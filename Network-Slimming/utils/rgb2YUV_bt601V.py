import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
from dataloader_mulit_patch import get_sub_folders
from yuv_rgb_convert import rgb2yuv444_bt601_video_range

def get_recursive_file_list(path, file_lst, ext):
    current_files = os.listdir(path)
    for file_name in current_files:
        full_file_name = path + "/" + file_name
        if os.path.isdir(full_file_name):
            get_recursive_file_list(full_file_name, file_lst, ext)
        elif full_file_name[-len(ext):]==ext:
            file_lst.append(full_file_name)
        else:
            None

def find_miss():
    with open('./train_list.txt','r') as f:
        lines = f.readlines()
        root_dir = [i.strip('\n') for i in lines]

    sub_folders = []
    get_sub_folders(root_dir, sub_folders)
    sub_folders.sort()
    not_conclude_YUV = []
    for sub_folder in tqdm(sub_folders):
        img_lst = []
        get_recursive_file_list(sub_folder, img_lst, ".jpg")
        
        if len(img_lst) % 2 != 0:
            not_conclude_YUV.append(os.path.dirname(img_lst[0]))
            continue
        # for img_path in img_lst:
    print(not_conclude_YUV)

def check_filenum_by_type():
    with open('./train_list.txt','r') as f:
        lines = f.readlines()
        root_dir = [i.strip('\n') for i in lines]
    
    for sub_folder in root_dir: #tqdm(root_dir):
        npy_num = 0
        jpg_num = 0
        yuv_num = 0
        img_lst = []
        get_recursive_file_list(sub_folder, img_lst, ".jpg")

        for img_path in img_lst:
            if '_YUV_bt601V' in img_path:
                yuv_num += 1
            elif '.npy' in img_path:
                npy_num += 1
            elif '.jpg' in img_path:
                jpg_num += 1
        print('yuv_num =', yuv_num, 'npy_num =', npy_num, 'jpg_num =', jpg_num)

def check_npy():
    with open('./train_list.txt','r') as f:
        lines = f.readlines()
        root_dir = [i.strip('\n') for i in lines]
    
    for sub_folder in root_dir: #tqdm(root_dir):
        img_lst = []
        get_recursive_file_list(sub_folder, img_lst, ".npy")
        print('npy_num =', len(img_lst))

def remove_npy():
    with open('./train_list.txt','r') as f:
        lines = f.readlines()
        root_dir = [i.strip('\n') for i in lines]
    count = 0
    for sub_folder in root_dir: #tqdm(root_dir):
        img_lst = []
        get_recursive_file_list(sub_folder, img_lst, ".npy")
        for img_path in tqdm(img_lst):
            count += 1
            os.remove(img_path)
    print(count)

def remove_duplicate():
    with open('./train_list.txt','r') as f:
        lines = f.readlines()
        root_dir = [i.strip('\n') for i in lines]

    sub_folders = []
    get_sub_folders(root_dir, sub_folders)
    sub_folders.sort()
    count = 0
    for sub_folder in tqdm(sub_folders):
        img_lst = []
        get_recursive_file_list(sub_folder, img_lst, ".jpg")

        for img_path in img_lst:
            if '_YUV_bt601V_YUV_bt601V' in img_path:
                os.remove(img_path)
                count += 1
            if '_YUV_bt601V' in img_path:
                os.remove(img_path)
                count += 1
    print(count)

def gen_yuv():
    # with open('./train_list.txt','r') as f:
    #     lines = f.readlines()
    #     root_dir = [i.strip('\n') for i in lines]

    # sub_folders = []
    # get_sub_folders(root_dir, sub_folders)
    # sub_folders.sort()

    root_dirs = ['/data2/myxu/tsr-classify-training/home.bak/nv3070/ljj_project/DATA/奇瑞问题压制']
    cmd_str = 'ls -lR ' + root_dirs[0] + ' | grep "^-" | wc -l'
    string = os.popen(cmd_str).read().strip()
    if string != '':
        print(int(string))
    for folder in tqdm(root_dirs):
        img_lst = []
        get_recursive_file_list(folder, img_lst, ".jpg")

        for img_path in tqdm(img_lst):
            if '_YUV_bt601V' in img_path:
                continue
            # img = img.transpose(2, 0, 1)
            dirpath = os.path.dirname(img_path)
            if len(os.path.basename(img_path).split('.')) == 2:
                basename = os.path.basename(img_path).split('.')[0] + '_YUV_bt601V.jpg'
            else:# 有的文件名中含有. 需要分割再拼接
                basename = ''
                for seg in os.path.basename(img_path).split('.')[:-1]:
                    basename += seg
                    basename += '.'
                basename = basename[:-1] + '_YUV_bt601V.jpg'
            save_path = dirpath+'/'+basename
            if not os.path.exists(save_path):
                img = cv2.imread(img_path).astype(np.uint8)
                img = img[:,:,::-1]
                img = rgb2yuv444_bt601_video_range(img, img.shape[1], img.shape[0])
                cv2.imwrite(save_path, img)

    string = os.popen(cmd_str).read().strip()
    if string != '':
        print(int(string))
        
if __name__ == '__main__':
    mode = 'gen'

    if mode == 'gen':
        gen_yuv()
    elif mode == 'remove':
        remove_duplicate()
    elif mode == 'miss':
        find_miss()
    elif mode == 'check':
        check_filenum_by_type()
    elif mode == 'check_npy':
        check_npy()
    elif mode == 'remove_npy':
        remove_npy()