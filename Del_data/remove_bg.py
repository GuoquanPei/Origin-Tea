'''
轮廓切割算法消除背景,将去除背景的图片存入到另一个文件夹中
'''
import os
import cv2
import numpy as np

def Remove_background(image_path, output_path):
    '''
    :param image_path:图像的路径
    :param output_path: 保存裁剪图片的文件夹
    :return:
    '''

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 创建掩膜
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)

        # 应用掩膜
        result = cv2.bitwise_and(image, image, mask=mask)
    else:
        print("No contours found.")
    # 显示结果
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(output_path, result)

def main():
    input_folder = r'E:\peiguoquan_data\Tea_Origin\bahaoshuiku'
    output_folder = r'E:\peiguoquan_data\Tea_origin_remove_bg'

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.JPG') or filename.endswith('.png'):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # 处理图像并保存
            Remove_background( image_path, output_path)


if __name__ == '__main__':
    main()
