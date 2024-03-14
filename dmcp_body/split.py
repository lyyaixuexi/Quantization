import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_folder, output_folder, test_size=0.2, random_state=42):
    # 获取所有类别文件夹的列表
    categories = os.listdir(input_folder)

    # 遍历每个类别文件夹
    for category in categories:
        category_path = os.path.join(input_folder, category)

        # 获取该类别下所有图片的文件列表
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]

        # 将图片列表划分为训练集和测试集
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

        # 创建输出文件夹中的对应类别子文件夹
        train_category_path = os.path.join(output_folder, 'train', category)
        test_category_path = os.path.join(output_folder, 'test', category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # 复制训练集图片到对应类别文件夹
        for image in train_images:
            src_path = os.path.join(category_path, image)
            dst_path = os.path.join(train_category_path, image)
            shutil.copyfile(src_path, dst_path)

        # 复制测试集图片到对应类别文件夹
        for image in test_images:
            src_path = os.path.join(category_path, image)
            dst_path = os.path.join(test_category_path, image)
            shutil.copyfile(src_path, dst_path)

if __name__ == "__main__":
    input_folder = "/mnt/cephfs/home/lyy/data/body_detect/reco_data"
    output_folder = "/mnt/cephfs/home/lyy/data/body_detect/split_data"

    split_dataset(input_folder, output_folder, test_size=0.2, random_state=42)
