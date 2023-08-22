import glob
import cv2 
from tqdm import tqdm
import os 


def one():
    package1 = os.listdir('/home/nv3070/ljj_project/DATA/classification_80/')
    package2 = os.listdir('/home/nv3070/ljj_project/DATA/电子限速牌分类_ljj/')

    count = 0
    for package in tqdm(package1):

        if package not in package2:
            os.makedirs('/home/nv3070/ljj_project/DATA/电子限速牌分类_ljj/{}'.format(package))

    print(count)

def two():
    for root,_,filelist in os.walk('/home/nv3070/ljj_project/DATA/电子限速牌分类_ljj/'):
        