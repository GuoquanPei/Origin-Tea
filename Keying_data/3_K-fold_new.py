import os
import random
from collections import defaultdict


def k_fold_split_stratified_by_region(input_file, output_dir='kfold_splits', k=10,
                                      train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按地区字段分层抽样的K折交叉验证划分
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # Step 1. 按地区字段分组
    region_groups = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 提取地区字段，假设路径中形式为 \地区\ 文件夹
            parts = line.split('\\')
            if len(parts) >= 2:
                region = parts[-2]  # 倒数第二个字段就是地区
            else:
                region = 'unknown'
            region_groups[region].append(line + '\n')

    print(f"共有 {len(region_groups)} 个地区。")

    # Step 2. 对每个地区的样本打乱并分K折
    folds = [[] for _ in range(k)]
    random.seed(seed)
    for region, samples in region_groups.items():
        random.shuffle(samples)
        n = len(samples)
        fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]  # 平均分折
        start = 0
        for i in range(k):
            end = start + fold_sizes[i]
            folds[i].extend(samples[start:end])
            start = end

    # Step 3. 为每一折生成 train/val/test
    os.makedirs(output_dir, exist_ok=True)
    for i in range(k):
        test_set = folds[i]
        remaining = [item for j, f in enumerate(folds) if j != i for item in f]
        random.shuffle(remaining)

        total_remaining = len(remaining)
        train_count = int(total_remaining * (train_ratio / (train_ratio + val_ratio)))
        train_set = remaining[:train_count]
        val_set = remaining[train_count:]

        # 保存
        fold_dir = os.path.join(output_dir, f'fold_{i + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'train.txt'), 'w', encoding='utf-8') as f_train:
            f_train.writelines(train_set)
        with open(os.path.join(fold_dir, 'val.txt'), 'w', encoding='utf-8') as f_val:
            f_val.writelines(val_set)
        with open(os.path.join(fold_dir, 'test.txt'), 'w', encoding='utf-8') as f_test:
            f_test.writelines(test_set)

        print(f"✅ Fold {i + 1}: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    print(f"\n✅ 完成 {k} 折按地区分层交叉验证划分，数据保存在 '{output_dir}'")


# 示例用法
if __name__ == '__main__':
    k_fold_split_stratified_by_region(
        input_file=r'Keying_data.txt',
        output_dir='kfold_splits',
        k=10,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=2025
    )
