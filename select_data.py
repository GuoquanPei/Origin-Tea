# -*- coding: utf-8 -*-
# @Author  : NaiChuan
# @Time    : 2026/1/26 14:10
# @File    : select.py
# @Software: PyCharm
import os
import random
import shutil
from pathlib import Path


def split_dataset_stratified(source_dir, target_dir, ratio=0.3):
    """
    对数据集进行分层随机抽样。

    Args:
        source_dir (str): 原始数据文件夹路径 (例如 'data')
        target_dir (str): 新数据文件夹路径 (例如 'new_data')
        ratio (float): 抽样比例 (0.3 表示 30%)
    """

    # 确保源目录存在
    if not os.path.exists(source_dir):
        print(f"错误：找不到源目录 {source_dir}")
        return

    # 如果目标目录不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")
    else:
        print(f"警告：目标目录 {target_dir} 已存在，新文件将合并进去。")

    # 获取所有类别（即子文件夹名称）
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    total_files_copied = 0

    print(f"开始处理，共发现 {len(classes)} 个类别...")
    print("-" * 30)

    for class_name in classes:
        # 构建当前类别的源路径和目标路径
        class_source_path = os.path.join(source_dir, class_name)
        class_target_path = os.path.join(target_dir, class_name)

        # 获取该类别下所有文件
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        num_files = len(files)

        # 计算需要抽取的数量
        sample_size = int(num_files * ratio)

        # 如果文件太少（比如只有1-2个），确保至少抽1个，或者根据需求调整策略
        if num_files > 0 and sample_size == 0:
            sample_size = 1

        # 随机抽样
        sampled_files = random.sample(files, sample_size)

        # 创建对应的目标子文件夹
        os.makedirs(class_target_path, exist_ok=True)

        # 复制文件
        for file_name in sampled_files:
            src_file = os.path.join(class_source_path, file_name)
            dst_file = os.path.join(class_target_path, file_name)
            shutil.copy2(src_file, dst_file)  # copy2 保留文件元数据（如创建时间）

        total_files_copied += len(sampled_files)
        print(
            f"类别 '{class_name}': 原有 {num_files} -> 抽取 {len(sampled_files)} ({(len(sampled_files) / num_files) * 100:.1f}%)")

    print("-" * 30)
    print(f"处理完成！共复制了 {total_files_copied} 个文件到 '{target_dir}' 下。")


# --- 配置部分 ---
if __name__ == "__main__":
    # 根据你的截图，建议将此脚本放在 'Tea_Origin' 目录下运行

    # 原始数据文件夹
    SOURCE_DATA_DIR = 'Scan_data'

    # 新数据文件夹
    NEW_DATA_DIR = 'new_Scan_data'

    # 抽样比例
    SAMPLE_RATIO = 0.3

    # 为了结果可复现，可以设置随机种子（可选）
    # random.seed(42)

    split_dataset_stratified(SOURCE_DATA_DIR, NEW_DATA_DIR, SAMPLE_RATIO)