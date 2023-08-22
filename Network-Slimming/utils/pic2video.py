import cv2 
import numpy as np 

import os 
from os.path import join
from tqdm import tqdm

def get_key(name):
    num = name.split('.')[0]
    num = int(num)
    return num



def jpg2video(package_path,save_path):

    img_list = os.listdir(package_path)
    img_list.sort(key=get_key)

    video = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'MJPG'),25,(1920,1080))

    for i in tqdm(img_list):
        img = cv2.imread(join(package_path,i))
        img = cv2.resize(img,(1920,1080))
        video.write(img)
    video.release()


if __name__ == '__main__':
    path = '/home/nv3070/ljj_project/CODE/视频样本挖掘集合/mdc_问题反馈/video_output/20220915/tsr'
    name_list = os.listdir(path)
    for i in name_list:
        package_path = join(path,i)
        save_path = join(path,i+'.avi')
        jpg2video(package_path,save_path)