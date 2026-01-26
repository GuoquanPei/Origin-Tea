import csv
from collections import defaultdict
import os

def count_disease_labels(txt_file, output_csv):
    label_counts = defaultdict(int)

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            path = parts[0]
            # 提取data之后的文件夹名作为类别
            try:
                category = path.split("Lincang-RGB-cut-power\\")[1].split("\\")[0]
                category_clean = category.replace("___", " ").replace("_", " ")
                label_counts[category_clean] += 1
            except IndexError:
                continue

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["类别", "数量"])
        for category, count in sorted(label_counts.items()):
            writer.writerow([category, count])

    print(f"统计结果已保存至：{output_csv}")

# 替换为你的文件路径
txt_file_path = "Lincang-RGB-cut-power.txt"
output_csv_path = "label_counts.csv"

count_disease_labels(txt_file_path, output_csv_path)
