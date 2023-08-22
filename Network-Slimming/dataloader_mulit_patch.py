import os
import collections
from tensorboardX import writer
import torch
import torchvision
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from utils.yuv_rgb_convert import rgb2yuv444_bt601_video_range
from os.path import join
import random
import cv2

##------------bgr 2 yuv420sp------------##

def bgr2420sp(bgr):
    yuv420 = cv2.cvtColor(bgr, cv2.COLOR_BGRA2YUV_I420)
    h, w = yuv420.shape
    y_pos = h // 3 * 2
    uv_length = h // 6
    img = np.zeros(shape=(h, w), dtype=np.uint8)

    u_plane = yuv420[y_pos:y_pos + uv_length, :]
    v_plane = yuv420[y_pos + uv_length:, :]
    img[:y_pos, :] = yuv420[:y_pos, :]
    img[y_pos:, ::2] = np.reshape(u_plane, newshape=(uv_length * 2, w // 2))
    img[y_pos:, 1::2] = np.reshape(v_plane, newshape=(uv_length * 2, w // 2))

    return img

##------------bgr 2 y_u_v ------------##
def uint8_bgr_to_yuv_separately(bgr):
    """
    Parameters
    ----------
    bgr input image: 8-bit unsigned
    Returns y,u,v images with same type as bgr
    -------
    """
    assert bgr.dtype == np.uint8

    data_type = bgr.dtype
    yuv420 = cv2.cvtColor(bgr, cv2.COLOR_BGRA2YUV_I420)
    h, w = yuv420.shape
    y_pos = h // 3 * 2

    uv_height = h // 6
    uv_width = w // 2

    y_plane = yuv420[:y_pos, :]
    u_plane = yuv420[y_pos:y_pos + uv_height, :]
    v_plane = yuv420[y_pos + uv_height:, :]

    u_plane = np.reshape(u_plane, newshape=(uv_height * 2, w // 2))
    v_plane = np.reshape(v_plane, newshape=(uv_height * 2, w // 2))

    ## 拼成y在上uv在下的形式
    uv_plane = np.concatenate((u_plane,v_plane),axis=1)
    yuv_plane = np.concatenate((y_plane,uv_plane),axis=0)
    # return y_plane, u_plane, 
    return yuv_plane

def check_if_contain_YUV_bt601V(path, contain_lst, question_dir, ext):
    current_files = os.listdir(path)
    for file_name in current_files:
        full_file_name = path + "/" + file_name
        if os.path.isdir(full_file_name):
            sub_contain_lst = []
            check_if_contain_YUV_bt601V(full_file_name, sub_contain_lst, question_dir, ext)
            if len(os.listdir(full_file_name)) != 0 and len(sub_contain_lst) == 0:
                print('no YUV_bt601V in this folder:', full_file_name)
                question_dir.append(full_file_name)
                contain_lst = []
                return 0
            else:
                contain_lst.extend(sub_contain_lst)
        elif full_file_name[-len(ext):]==ext:
            if 'YUV_bt601V' in full_file_name:
                contain_lst.append(full_file_name)
                return 0
        else:
            None

def get_recursive_file_list(path, file_lst, color_mode, ext):
    current_files = os.listdir(path)
    for file_name in current_files:
        full_file_name = path + "/" + file_name
        if os.path.isdir(full_file_name):
            get_recursive_file_list(full_file_name, file_lst, color_mode, ext)
        elif full_file_name[-len(ext):]==ext:
            if color_mode == 'YUV_bt601V' and color_mode in full_file_name:
                file_lst.append(full_file_name)
            elif color_mode in ['bgr', 'yuv420sp'] and 'YUV_bt601V' not in full_file_name:
                file_lst.append(full_file_name)
        else:
            None

# Get the all sub folders in the "path" folder
def get_sub_folders(path, sub_folder):

    if not type(path) == type(list()):
        path = [path]

    for each_path in path :
        current_files = os.listdir(each_path)
        for file_name in current_files:
            full_file_name = os.path.join(each_path, file_name)
            if os.path.isdir(full_file_name):
                sub_folder.append(full_file_name)
            else:
                None

class img_cls_by_dir_loader(torch.utils.data.Dataset):
    def __init__(self, root_dir,  split="train", is_transform=True, img_size=[48, 48],color_mode = 'bgr', multi_patch = True):
        self.color_mode = color_mode
        # sub_folders = []
        # get_sub_folders(root_dir, sub_folders)
        # sub_folders.sort()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.multi_patch = multi_patch

        self.labels = dict()
        self.labellist = self.get_train_label_list()
        self.files = dict()
        self.files_img = dict()

        file_lst = []
        print("there are ", len(self.labellist), " classes")
        self.n_classes = len(self.labellist)

        fail_lst = []
        contain_lst = []
        question_dir = []
        for data_dir in self.root_dir:
            if self.color_mode == 'YUV_bt601V':
                check_if_contain_YUV_bt601V(data_dir, contain_lst, question_dir, ".jpg")
                if len(question_dir) != 0:
                    print(data_dir)
                    print("文件夹中没有YUV_bt601V图片,请转换后再训练！")
                    exit(-1)

            img_lst = []
            get_recursive_file_list(data_dir, img_lst, self.color_mode, ".jpg")
            for img_path in img_lst:
                temp_label = img_path.split('/')
                try:
                    self.labels[img_path] = self.labellist.index(temp_label[-2])
                except:
                    try:
                        self.labels[img_path] = self.labellist.index(temp_label[-3])
                    except:
                        try:
                            self.labels[img_path] = self.labellist.index(temp_label[-4])
                        except:
                            try:
                                self.labels[img_path] = self.labellist.index(temp_label[-5])
                            except:
                                fail_lst.append(img_path)
                                continue
                file_lst.append(img_path)

        random.seed(42)
        random.shuffle(file_lst)
        self.files["train"] = file_lst[0:int(len(file_lst)*0.99)]
        self.files["val"] = file_lst[int(len(file_lst) * 0.99):]
        self.files["train+val"] = file_lst
        idx = 0

        self.indices = range(len(self.files[self.split])) ## number of pic
        self.n = len(self.files[self.split])
        self.epoch = 'epoch_0'

    def get_train_label_list(self):
        path = './small_sign_classify_class_list.txt'
        labellist = list()
        with open(path,'r') as f:
            for line in f:
                i = line.strip('\n')
                labellist.append(i)
        return labellist

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path= self.files[self.split][index]
        #lbl = self.labels[img_path]

        ## 从内存中拿数据
        #img = self.files_img[img_path]
        ## yuv图像作为网络输入
        #img = bgr2420sp(img)

        ## 如果采用多patch组合策略
        if self.multi_patch:
            img = np.zeros([288,288,3])
            label = np.zeros([3,3],dtype=np.float32)
            indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(8)]

            ## 这里相当于一拖8，做一个随机拼接图像的操作，完成随机输入
            for i, index in enumerate(indices):
                img_path= self.files[self.split][index]
                temp_img = cv2.imread(img_path)
                temp_img = cv2.resize(temp_img,(96,96))

                if i == 0:#放中间
                    img[96*1:96*1+96,96*1:96*1+96,:] = temp_img
                    label[1,1] = self.labels[img_path]
                elif i == 1:# 一号位
                    img[96*0:96*0+96,96*0:96*0+96,:] = temp_img
                    label[0,0] = self.labels[img_path]
                elif i == 2:# 二号位
                    img[96*0:96*0+96,96*1:96*1+96,:] = temp_img
                    label[0,1] = self.labels[img_path]
                elif i == 3:#
                    img[96*0:96*0+96,96*2:96*2+96,:] = temp_img
                    label[0,2] = self.labels[img_path]
                elif i == 4:
                    img[96*1:96*1+96,96*0:96*0+96,:] = temp_img
                    label[1,0] = self.labels[img_path]
                elif i == 5:
                    img[96*1:96*1+96,96*2:96*2+96,:] = temp_img
                    label[1,2] = self.labels[img_path]
                elif i == 6:
                    img[96*2:96*2+96,96*0:96*0+96,:] = temp_img
                    label[2,0] = self.labels[img_path]
                elif i == 7:
                    img[96*2:96*2+96,96*1:96*1+96,:] = temp_img
                    label[2,1] = self.labels[img_path]
                elif i == 8:
                    img[96*2:96*2+96,96*2:96*2+96,:] = temp_img
                    label[2,2] = self.labels[img_path]
        else:
            img_path= self.files[self.split][index]
            img = cv2.imread(img_path)
            img = cv2.resize(img,(self.img_size, self.img_size))
            label = self.labels[img_path]
            label = np.array(label)

        if self.is_transform:
            img = self.transform(img.astype(np.uint8))


        # 这里由于yuv444在线处理过慢，导致的
        # if not os.path.exists(join(self.savepath,self.epoch,'img')):
        #     os.makedirs(join(self.savepath,self.epoch,'img'))
        # if not os.path.exists(join(self.savepath,self.epoch,'label')):
        #     os.makedirs(join(self.savepath,self.epoch,'label'))

        # ## yuv as jpg,label as npy
        # cv2.imwrite(join(self.savepath,self.epoch,'img','%d.jpg' % index),img)
        # np.save(join(self.savepath,self.epoch,'label','%d.npy' % index),label)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()
        return img, label

    def transform(self, img):
        if self.color_mode == 'yuv420sp':
            img= bgr2420sp(img)
            img = np.expand_dims(img, 0)
        elif self.color_mode == 'y_u_v':
            img= uint8_bgr_to_yuv_separately(img)
            img = np.expand_dims(img, 0)
        elif self.color_mode == 'bgr':
            ## NHWC -> NCHW
            img = img.transpose(2, 0, 1)
        elif self.color_mode == 'rgb':
            img = img[:,:,::-1]
            img = img.transpose(2, 0, 1)
        elif self.color_mode == 'yuv444':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img.transpose(2, 0, 1)

        elif self.color_mode == 'YUV_bt601V':
            # img = img[:,:,::-1]
            # img = rgb2yuv444_bt601_video_range(img, img.shape[1], img.shape[0])
            img = img.transpose(2, 0, 1)
        else:
            print('wrong color mode,check u choose in ["yuv","bgr","rgb","y_u_v","yuv444"]')
        img = img.astype(float) / 256.0

        return img

class dataset_preprocessed(torch.utils.data.Dataset):

    def __init__(self,data_path) -> None:
        super(dataset_preprocessed).__init__()
        self.data_path = data_path
        self.n_classes = 279
        self.trainpathlist = self.get_files_list()

    def get_files_list(self):
        pathlist = []
        for i in os.listdir(join(self.data_path,'img')):
            pathlist.append(join(self.data_path,'img',i))
        return pathlist
    
    def __len__(self):
        return len(self.trainpathlist)

    def __getitem__(self, index):
        img_path = self.trainpathlist[index]
        label_path = img_path.replace('img','label').replace('.jpg','.npy')

        img = cv2.imread(img_path)
        img = img.transpose(2, 0, 1)
        img = img.astype(float) / 256.0

        label = np.load(label_path)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()
        return img, label

def main():
    path = '/home.bak/nv3070/ljj_project/DATA/小标牌YUV601Video增广数据存储/epoch_0'
    dataset = dataset_preprocessed(path)
    trainloader = data.DataLoader(
            dataset, batch_size=128, num_workers=4, shuffle=True,pin_memory=True
        )
    for epoch in range(20):
        for i,input_data in enumerate(trainloader):
            print('\r process epoch:{},iter:{} trainloader_length:{} finish:{}%'.format(epoch,i,len(trainloader),i/len(trainloader)*100),end='')



def old_test():
    from tqdm import tqdm
    import torchvision
    ## check the dataloader 
    from tensorboardX import SummaryWriter 
    writer = SummaryWriter()

    ## root_dir应该是个list
    root_dir = [        
        "/home.bak/nv3070/ljj_project/DATA/classification_80",
        "/home.bak/nv3070/ljj_project/DATA/电子限速牌分类_ljj",
        "/home.bak/nv3070/ljj_project/DATA/G1S限重标牌压制数据",
        "/home.bak/nv3070/ljj_project/DATA/qirui_20220622_small_crop_diff",
        "/home.bak/nv3070/ljj_project/DATA/zxd_tsr_补充训练数据",
        "/home.bak/nv3070/ljj_project/DATA/qirui_20220805",]

    # root_dir = ["/home.bak/nv3070/ljj_project/DATA/qirui_20220805",]
    cls_lst = './test_ljj.txt'
    dst = img_cls_by_dir_loader(
        root_dir,
        cls_lst,
        split="train",
        is_transform=True,
        color_mode="YUV_bt601V",
    )

    labellist = dst.get_train_label_list()


    for epoch in range(20):
        dst.epoch = 'epoch_{}'.format(epoch)
        trainloader = data.DataLoader(
            dst, batch_size=128, num_workers=4, shuffle=True,pin_memory=True
        )
        for i, input_data in enumerate(dst):
            print('\r process epoch:{},iter:{} trainloader_length:{} finish:{}%'.format(epoch,i,len(dst),i/len(dst)*100),end='')


if __name__ == "__main__":
    main()