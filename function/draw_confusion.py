'''
这个函数用来绘制confusion矩阵图，并保存
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#绘制热力图并进行保存
def draw_heatmap(data, xticklabels, yticklabels, title, save_path):
    fig, ax = plt.subplots(figsize=(30, 30))  # 设置整个图表的尺寸
    # 指定字体文件
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置支持中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 创建一个颜色映射对象，用于热力图的颜色映射
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # 使用seaborn的heatmap函数创建相关性热力图
    sns.heatmap(data, cmap=cmap, annot=True, fmt=".0f",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"fontsize":30})
    #设置热力条刻度字体
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=40)  # colorbar 刻度字体大小
    # 设置坐标轴和标题字体大小
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # 设置图表标题
    plt.title(title,fontsize=70)
    # 保存，save要在show之前
    plt.savefig(save_path, dpi=200)
    # 显示图表
    plt.show()

def draw_cong_mat(data, num_class, title = '混淆矩阵', save_path = 'D:\DIP\\results\Confusion_hotmap.png'):
    class_list = [str(i) for i in range(1, num_class+1)]  # 假设你想要生成到数字10
    draw_heatmap(data, class_list, class_list, title, save_path)

if __name__ == '__main__':
    a1 = np.random.rand(34, 34)
    draw_cong_mat(a1, num_class=34)
