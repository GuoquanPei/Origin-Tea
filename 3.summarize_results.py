'''
用于统计实验结果，输出mean或mean±std
'''

import os
import re
import pandas as pd
import numpy as np


def extract_metrics(report_path):
    """从final_test_report.txt中提取OA, AA, Kappa和F1"""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取OA
    oa_match = re.search(r'Final整体测试集上的OA:\s+([0-9.]+)', content)
    oa = float(oa_match.group(1)) if oa_match else None

    # 提取AA
    aa_match = re.search(r'Final整体测试集上的AA:\s+([0-9.]+)', content)
    aa = float(aa_match.group(1)) if aa_match else None

    # 提取Kappa
    kappa_match = re.search(r'Final整体测试集上的kappa系数:\s+([0-9.]+)', content)
    kappa = float(kappa_match.group(1)) if kappa_match else None

    # 提取weighted avg f1-score
    f1_match = re.search(r'weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', content)
    f1 = float(f1_match.group(1)) if f1_match else None

    # 提取weighted avg precision
    precision_match = re.search(r'weighted avg\s+([\d.]+)', content)
    precision = float(precision_match.group(1)) if precision_match else None

    # 提取weighted avg recall
    recall_match = re.search(r'weighted avg\s+[\d.]+\s+([\d.]+)', content)
    recall = float(recall_match.group(1)) if recall_match else None

    return oa, aa, kappa, precision, recall, f1


def main(results_root='results2', output_csv='summary_results.csv', output_mode='mean±std'):
    """
    统计实验结果，并输出为CSV。
    参数:
        results_root: 结果文件夹
        output_csv: 输出CSV路径
        output_mode: 'mean' 或 'mean±std'
    """
    assert output_mode in ['mean', 'mean±std'], "output_mode必须是'mean'或'mean±std'"

    summary = []

    for model_dir in os.listdir(results_root):
        model_path = os.path.join(results_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        # 初始化存储
        oa_list, aa_list, kappa_list, precision_list, recall_list, f1_list = [], [], [], [], [], []

        for fold in os.listdir(model_path):
            fold_path = os.path.join(model_path, fold)
            report_path = os.path.join(fold_path, 'final_test_report.txt')
            if os.path.isfile(report_path):
                oa, aa, kappa, precision, recall, f1 = extract_metrics(report_path)
                if None not in (oa, aa, kappa, precision, recall, f1):
                    oa_list.append(oa)
                    aa_list.append(aa)
                    kappa_list.append(kappa)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

        if len(oa_list) == 0:
            print(f"Warning: {model_dir} has no valid reports.")
            continue

        row = {'模型': model_dir}
        metrics = {
            'OA': oa_list,
            'AA': aa_list,
            'KAPPA': kappa_list,
            'Precision': precision_list,
            'Recall': recall_list,
            'F1': f1_list
        }

        for metric_name, values in metrics.items():
            mean = np.mean(values)
            if output_mode == 'mean±std':
                std = np.std(values, ddof=1)
                row[metric_name] = f"{mean:.4f} ± {std:.4f}"
            else:  # output_mode == 'mean'
                row[metric_name] = f"{mean:.4f}"

        summary.append(row)

    # 保存CSV
    df = pd.DataFrame(summary)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"汇总结果已保存至: {output_csv}")


if __name__ == "__main__":
    #  output_mode=['mean', 'mean±std']
    main(results_root='Scan_training', output_csv='Scan_training/summary_results.csv', output_mode='mean')
