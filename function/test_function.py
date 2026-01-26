import torch
import numpy as np
from sklearn import metrics


def test(args, iter, test_loader, model):
    model.eval()
    predictions = torch.FloatTensor().to(args.device)
    labels = torch.FloatTensor().to(args.device)

    with torch.no_grad():
        for (patches, targets) in test_loader:
            patches = patches.to(args.device)
            targets = targets.to(args.device)
            output = model(patches)
            batch_prediction = output.argmax(1)
            predictions = torch.cat((predictions, batch_prediction), 0)
            labels = torch.cat((labels, targets), 0)

    predictions = predictions.cpu()
    labels = labels.cpu()

    # 计算混淆矩阵
    conf_mat = metrics.confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(conf_mat)

    print("------整体验证集上的报告------")
    report = metrics.classification_report(labels, predictions, digits=4)
    acc_for_each_class = metrics.precision_score(labels, predictions, average=None)
    test_OA = metrics.accuracy_score(labels, predictions)
    test_AA = np.mean(acc_for_each_class)
    test_kappa = metrics.cohen_kappa_score(labels, predictions)

    # ✅ 计算 MCC（多分类支持）
    test_MCC = metrics.matthews_corrcoef(labels, predictions)

    print(f"整体测试集上的OA: {test_OA:.4f}")
    print(f"整体测试集上的AA: {test_AA:.4f}")
    print(f"整体测试集上的Kappa: {test_kappa:.4f}")
    print(f"整体测试集上的MCC: {test_MCC:.4f}")  # ✅ 新增
    print(f"test_OA={round(test_OA, 4)}")

    # ✅ 返回值中增加 test_MCC
    return test_OA, test_AA, test_kappa, test_MCC, report, conf_mat
