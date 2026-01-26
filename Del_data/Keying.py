import cv2
import numpy as np
import os
from tqdm import tqdm

input_root = 'E:\peiguoquan_data\Tea_Origin\Tea_Origin'  # 注意路径斜杠方向
output_root = 'E:/peiguoquan_data/Tea_origin_keying'
os.makedirs(output_root, exist_ok=True)

# 茶叶颜色极限范围 (基于提供的RGB数据)
lower = np.array([20, 30, 0])    # 更低更宽，避开背景颜色
upper = np.array([170, 180, 120]) # 更高更宽，确保覆盖所有茶叶颜色但不包括背景

for subdir, dirs, files in os.walk(input_root):
    for file in tqdm(files):
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG')):
            file_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(subdir, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            # 安全读取
            with open(file_path, 'rb') as f:
                img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if img is None:
                print(f"无法读取图像: {file_path}")
                continue

            # 转 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 掩膜
            mask = cv2.inRange(img_rgb, lower, upper)

            # 可选：平滑边界 (形态学操作)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去杂点

            # 三通道掩膜
            mask_3c = cv2.merge([mask, mask, mask])

            # 应用掩膜
            result = cv2.bitwise_and(img, mask_3c)

            # 背景设黑
            result[mask == 0] = [0, 0, 0]

            # 保存
            output_path = os.path.join(output_dir, file)
            cv2.imwrite(output_path, result)
