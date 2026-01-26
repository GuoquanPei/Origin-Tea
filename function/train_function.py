import os
import csv
import logging
import re
import time

import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from .test_function import test
from .val_function import validate


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))


def train_epoch(args, train_loader, model, criterion, optimizer, losses):
    """单轮训练过程"""
    model.train()
    scaler = GradScaler()  # 混合精度训练

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        with autocast():  # 自动混合精度上下文
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

        if step % 100 == 0:  # 每100 step打印一次损失
            print('| train loss: %.6f' % loss.item())

    return model


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_num, save_dir):
    """主训练过程"""
    best_OA = 0
    best_model = None
    losses = []
    val_OAs = []

    # 确保结果保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epoch_num)):
        logging.info(f"Epoch {epoch + 1}/{epoch_num} - Training...")
        train_epoch(args, train_loader, model, criterion, optimizer, losses)
        scheduler.step()

        # === 验证阶段 ===
        try:
            val_OA, val_AA, val_kappa, val_MCC, report, conf_mat = test(args, epoch, val_loader, model)
            val_OAs.append(val_OA)
            logging.info(
                f"Epoch {epoch + 1}: val_OA={val_OA:.4f}, val_AA={val_AA:.4f}, "
                f"Kappa={val_kappa:.4f}, MCC={val_MCC:.4f}"
            )
        except Exception as e:
            logging.error(f"Validation failed at epoch {epoch + 1}: {e}")
            continue  # 避免验证阶段异常终止训练

        # === 保存分类报告 ===
        if isinstance(report, np.ndarray):
            report = str(report)

        report_path = os.path.join(save_dir, f"report_epoch_{epoch + 1}.csv")
        try:
            with open(report_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["class", "precision", "recall", "f1-score", "support"])  # 表头

                # 写入各类别精度信息
                for line in report.split('\n')[2:-3]:
                    row = re.split(r'\s+', line.strip())
                    if len(row) >= 5 and row[0].replace('.', '', 1).isdigit():
                        writer.writerow(row)

                # 写入整体指标
                writer.writerow([])
                writer.writerow(["Overall Accuracy (OA)", val_OA])
                writer.writerow(["Average Accuracy (AA)", val_AA])
                writer.writerow(["Kappa", val_kappa])
                writer.writerow(["Matthews Corrcoef (MCC)", val_MCC])  # ✅ 新增

            logging.info(f"Saved classification report for epoch {epoch + 1}")

        except Exception as e:
            logging.error(f"Error saving report for epoch {epoch + 1}: {e}")

        # === 保存最优模型 ===
        if val_OA > best_OA:
            best_OA = val_OA
            best_model = model
            logging.info(f"New best model found at epoch {epoch + 1} with val_OA={val_OA:.4f}")

    return best_model, losses, val_OAs
