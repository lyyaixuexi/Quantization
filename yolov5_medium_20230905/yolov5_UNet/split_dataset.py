import os
import random
import shutil
# image_folder = "/mnt/cephfs/home/lyy/data/Traffic_detect/traffic_detect"  # 指定图像文件夹的路径
# image_files = os.listdir(image_folder)
#
# random.shuffle(image_files)  # 随机打乱图像文件列表
# split_index = 200
#
# val_split = image_files[:split_index]
# train_split = image_files[split_index:]
#
# # 创建第一个文本文件并写入图像文件名
# with open("/mnt/cephfs/home/lyy/data/Traffic_detect/traffic_detect/val_split.txt", "w") as file:
#     for image_file in val_split:
#         file.write(os.path.join(image_folder, image_file) + "\n")
#
# # 创建第二个文本文件并写入图像文件名
# with open("/mnt/cephfs/home/lyy/data/Traffic_detect/traffic_detect/train_split.txt", "w") as file:
#     for image_file in train_split:
#         file.write(os.path.join(image_folder, image_file) + "\n")




with open("/mnt/cephfs/home/lyy/data/traffic_detect/train.txt", "r") as file:
    lines = file.readlines()  # 读取文件的所有行
random.seed(42)
random.shuffle(lines)  # 随机打乱行的顺序

split_index = 200

os.makedirs("/mnt/cephfs/home/lyy/data/traffic_detect/images/train", exist_ok=True)
os.makedirs("/mnt/cephfs/home/lyy/data/traffic_detect/images/val", exist_ok=True)
os.makedirs("/mnt/cephfs/home/lyy/data/traffic_detect/labels/val", exist_ok=True)
os.makedirs("/mnt/cephfs/home/lyy/data/traffic_detect/labels/train", exist_ok=True)
# 第一部分
val_lines = lines[:split_index]
# with open("/mnt/cephfs/home/lyy/data/traffic_detect/val_split.txt", "w") as file:
for image_file in val_lines:
    image_info = image_file.split(' ')
    image_name = image_info[0][5:-4]
    boxes = image_info[1:]
    # print(boxes)
    # print(image_name)
    with open(f"/mnt/cephfs/home/lyy/data/traffic_detect/labels/val/{image_name}.txt", "w") as file:
        # 将每个边界框信息写入文件
        for box in boxes:
            box = box.replace('\n', '')
            info = box.split(',')
            info.insert(0, info.pop())
            info = ' '.join(info)
            print(info)
            file.write(info + '\n')


for line in val_lines:
    parts = line.strip().split(' ')
    image_path = parts[0]  # 图像路径
    image_path = os.path.join("/mnt/cephfs/home/lyy/data/traffic_detect", image_path)
    print(image_path)
    shutil.copy(image_path, "/mnt/cephfs/home/lyy/data/traffic_detect/images/val")

# # 第二部分
train_lines = lines[split_index:]

for image_file in train_lines:
    image_info = image_file.split(' ')
    image_name = image_info[0][5:-4]
    boxes = image_info[1:]
    # print(boxes)
    # print(image_name)
    with open(f"/mnt/cephfs/home/lyy/data/traffic_detect/labels/train/{image_name}.txt", "w") as file:
        # 将每个边界框信息写入文件
        for box in boxes:
            box = box.replace('\n', '')
            info = box.split(',')
            info.insert(0, info.pop())
            info = ' '.join(info)
            print(info)
            file.write(info + '\n')


for line in train_lines:
    parts = line.strip().split(' ')
    image_path = parts[0]  # 图像路径
    image_path = os.path.join("/mnt/cephfs/home/lyy/data/traffic_detect", image_path)
    print(image_path)
    shutil.copy(image_path, "/mnt/cephfs/home/lyy/data/traffic_detect/images/train")



