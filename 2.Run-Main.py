from my_dataset.my_dataset import CustomDataset



import argparse
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
from fvcore.nn import parameter_count_table
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import pandas as pd

from my_dataset.my_dataset_new import MyDataset
from function.train_function import train
from function.test_function import test
from function.draw_confusion import draw_cong_mat
from get_model_instance import get_model_instance

# ============ 解析参数 ============
parser = argparse.ArgumentParser("multi_model_kfold_training")
parser.add_argument('--exp_name', type=str, default='multi_model_experiment', help='experiment name')
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
parser.add_argument('--data_augmentation', type=bool, default=False, help='Whether or not to use data augmentation')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--class_num', type=int, default=7, help='Class number of tea_classification')
parser.add_argument('--folds', type=str, default='1-10', help='Which folds to run, e.g. "1,3,5" or "1-10"')
parser.add_argument('--models', type=str,
                    default=(
                        'coatnet_0,convnext_base,efficientnet_b0,ghostnetv3,'
                        'inceptionnext_small,resnet50,starnet_s1,mobilenetv4_small,'
                        'mobilenetv3_small,'
              
                        'mobilenetv3_tea_0mb,mobilenetv3_tea_1mb,mobilenetv3_tea_2mb,'
                        'mobilenetv3_tea_3mb,mobilenetv3_tea_4mb,mobilenetv3_tea_5mb,'
                        'mobilenetv3_tea_6mb,mobilenetv3_tea_7mb,mobilenetv3_tea_8mb,mobilenetv3_tea_9mb,'
                        'mobilenetv3_tea_s1_0mb,mobilenetv3_tea_s1_1mb,mobilenetv3_tea_s1_2mb,'
                        'mobilenetv3_tea_s1_3mb,mobilenetv3_tea_s1_4mb,mobilenetv3_tea_s1_5mb,'
                        'mobilenetv3_tea_s1_6mb,mobilenetv3_tea_s1_7mb,mobilenetv3_tea_s1_8mb,mobilenetv3_tea_s1_9mb,'
                        'mobilenetv3_tea_s1s2_0mb,mobilenetv3_tea_s1s2_1mb,mobilenetv3_tea_s1s2_2mb,'
                        'mobilenetv3_tea_s1s2_3mb,mobilenetv3_tea_s1s2_4mb,mobilenetv3_tea_s1s2_5mb,'
                        'mobilenetv3_tea_s1s2_6mb,mobilenetv3_tea_s1s2_7mb,mobilenetv3_tea_s1s2_8mb,'
                        'mobilenetv3_tea_s1s2_9mb'

                    ),
                    help='Comma-separated list of models to train')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ============ 解析折叠 ============
def parse_folds(fold_str):
    folds = []
    if '-' in fold_str:
        start, end = fold_str.split('-')
        folds = list(range(int(start), int(end) + 1))
    else:
        folds = [int(f.strip()) for f in fold_str.split(',')]
    return folds


# ============ 解析模型 ============
def parse_models(models_str):
    return [m.strip().lower() for m in models_str.split(',')]


# ============ 自定义DataLoaderX（背景加载） ============
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# ============ 优化器和调度器 ============
def get_optimizer_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))
    return optimizer, scheduler


# ============ 保存模型参数信息 ============
def save_model_params(model, results_dir):
    param_count = parameter_count_table(model)
    param_file = os.path.join(results_dir, 'model_params.txt')
    with open(param_file, 'w', encoding='utf-8') as f:
        f.write("==================== Model Parameter Count ====================\n")
        f.write(str(param_count))
    print(f'Model parameters saved to: {param_file}')


# ============ 运行单个模型（已共享数据） ============
def run_fold(model_name, fold_id, train_loader, val_loader, test_loader):
    print(f"\n=================== Model: {model_name.upper()} | Fold: {fold_id} ===================")

    # 1. 加载模型
    model = get_model_instance(model_name, args)
    model.eval()
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练")
        model = nn.DataParallel(model)

    # 2. 优化器、调度器、损失函数
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # 3. 结果保存路径
    results_dir = os.path.join('results_keying_data', f'{args.exp_name}_{model_name}', str(fold_id))
    os.makedirs(results_dir, exist_ok=True)

    # ============ 训练阶段 ============
    print('==================== 训练阶段 ====================')
    save_dir = os.path.join(results_dir, 'csv')
    start_time = time.time()
    model, losses, val_prces = train(args, model, train_loader, val_loader, criterion, optimizer, scheduler,
                                     args.epochs, save_dir)
    end_time = time.time()

    # ============ 测试阶段 ============
    print('==================== 测试阶段 ====================')
    OA, AA, kappa, MCC, report, confu_mat = test(args, 0, test_loader, model)

    # 保存模型参数信息
    save_model_params(model, results_dir)

    # 耗时
    print(f"Time taken in minutes: {(end_time - start_time) / 60:.2f} minutes")

    # ============ 结果可视化和保存 ============
    # 损失曲线
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_dir, 'losses_TL.png'), dpi=100)
    pd.DataFrame(losses, columns=['Loss']).to_excel(os.path.join(results_dir, 'losses_TL.xlsx'), index=False)
    plt.close()

    # 验证集精度曲线
    plt.plot(val_prces)
    plt.xlabel('Epoch')
    plt.ylabel('Val_OA')
    plt.savefig(os.path.join(results_dir, 'val_OA_TL.png'), dpi=300)
    pd.DataFrame(val_prces, columns=['Val_OA']).to_excel(os.path.join(results_dir, 'val_OA_TL.xlsx'), index=False)
    plt.close()

    # 混淆矩阵
    # draw_cong_mat(confu_mat, args.class_num, '混淆矩阵', results_dir)
    # ============ 保存混淆矩阵为 CSV 文件 ============
    cm_df = pd.DataFrame(confu_mat)
    cm_df.index = [f"True_{i}" for i in range(args.class_num)]
    cm_df.columns = [f"Pred_{i}" for i in range(args.class_num)]
    cm_path = os.path.join(results_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path, encoding='utf-8-sig')
    print(f"Confusion matrix saved to: {cm_path}")

    # 测试报告
    results_txt_path = os.path.join(results_dir, 'final_test_report.txt')
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write("==================Final 测试集报告==================\n")
        f.write(str(report) + '\n\n')
        f.write("Final整体测试集上的OA: {:.4f}\n".format(OA))
        f.write("Final整体测试集上的AA: {:.4f}\n".format(AA))
        f.write("Final整体测试集上的kappa系数: {:.4f}\n".format(kappa))
        f.write("Final整体测试集上的MCC: {:.4f}\n".format(MCC))
    print('测试集结果已保存至:', results_txt_path)

    # 保存模型权重
    save_path = os.path.join(results_dir, f'{model_name}_fold{fold_id}.pth')
    torch.save(model, save_path)
    print(f'模型已保存至: {save_path}')


# ============ 主函数 ============
def main():
    folds = parse_folds(args.folds)
    models = parse_models(args.models)
    for fold_id in folds:
        # ============ 同一折共享数据 ============
        train_txt = f'Keying_data/kfold_splits/fold_{fold_id}/train.txt'
        val_txt = f'Keying_data/kfold_splits/fold_{fold_id}/val.txt'
        test_txt = f'Keying_data/kfold_splits/fold_{fold_id}/test.txt'

        train_dataset = CustomDataset(train_txt,augment=True)
        train_loader = DataLoaderX(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   pin_memory=True)
        val_dataset = CustomDataset(val_txt,augment=False)
        val_loader = DataLoaderX(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 pin_memory=True)
        test_dataset = CustomDataset(test_txt,augment=False)
        test_loader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  pin_memory=True)

        for model_name in models:
            run_fold(model_name, fold_id, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()

