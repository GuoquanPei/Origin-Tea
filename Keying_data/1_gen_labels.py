from PIL import Image
import os
import random
from tqdm import tqdm


# 用于读取生成好的output.txt文件，按8：1：1的比例划分数据集
def get_train_val_test(output):
    # 从 output.txt 中读取图片路径和标签
    file_path = output
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 打乱数据
    random.shuffle(lines)

    # 计算划分的索引
    total_count = len(lines)
    train_count = int(0.8 * total_count)
    val_count = int(0.1 * total_count)
    test_count = total_count - train_count - val_count

    # 划分数据
    train_data = lines[:train_count]
    val_data = lines[train_count:train_count + val_count]
    test_data = lines[train_count + val_count:]

    # 写入到train.txt
    with open("train.txt", "w") as train_file:
        train_file.writelines(train_data)

    # 写入到val.txt
    with open("val.txt", "w") as val_file:
        val_file.writelines(val_data)

    # 写入到test.txt
    with open("test.txt", "w") as test_file:
        test_file.writelines(test_data)


# 定义图片大小的函数和检查图片异常并抛出
def check_image_open(image_path):
    try:
        # 尝试打开图像
        with Image.open(image_path):
            pass  # 如果成功，什么都不做
        return True  # 如果成功打开图像，则返回 True

    except Exception as e:
        print(f"打开图像时出错：{image_path}，{e}")
        return False  # 如果打开图像时出现错误，则返回 False


# 定义一个单独的函数用于重命名照片
def rename_images(subfolder_path):
    img_counter = 0  # 用于图片重命名
    for filename in tqdm(os.listdir(subfolder_path), desc="正在重命名图片"):
        image_path = os.path.join(subfolder_path, filename)

        if os.path.isfile(image_path) and check_image_open(image_path):
            new_image_name = f"{img_counter}.jpg"
            new_image_path = os.path.join(subfolder_path, new_image_name)
            os.rename(image_path, new_image_path)
            img_counter += 1


# 遍历文件夹并记录文件路径和标签
def process_folders(root_folder, output_file, rename=False):
    idx = 0  # 初始化标签,标签必须从0开始计数，不然会报错
    total_folders = sum(len(subfolders) for _, subfolders, _ in os.walk(root_folder))
    with open(output_file, 'w') as f:
        for folder_name, subfolders, _ in tqdm(os.walk(root_folder), total=total_folders, desc="正在处理文件夹"):
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_name, subfolder)
                if rename:
                    rename_images(subfolder_path)  # 调用重命名函数
                for filename in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, filename)

                    if os.path.isfile(image_path) and check_image_open(image_path):
                        f.write(f"{image_path} {idx}\n")
                idx += 1  # 增加标签以便下一个子文件夹使用


def main():
    # 指定根文件夹路径和输出文件名
    root_folder =  r'E:\peiguoquan_data\Tea_origin_keying'
    output_file = "Keying_data.txt"

    # 设置是否进行重命名,如果有中文路径的昵称就要重命名
    rename = False  # 将此值改为 True 或 False 控制是否重命名图片

    # 处理文件夹
    process_folders(root_folder, output_file, rename=rename)
    print('读取所有路径完毕')

    # 划分数据集
    get_train_val_test(output_file)


if __name__ == '__main__':
    main()
