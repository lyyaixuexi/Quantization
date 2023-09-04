from __future__ import print_function
import tflite_quantization_PACT_weight_and_act
import os
import json
import argparse
import shutil
import torch
import torch.nn as nn
from torch.utils import data
import models
import math
from apex import amp
from models.RanGer import Ranger
from models.Label_SmoothCELoss import LabelSmoothCELoss, LabelSmoothCELoss_modify
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataloader_mulit_patch import img_cls_by_dir_loader as data_loader
import numpy as np
import cv2
from lr_scheduling import *
from tqdm import tqdm
import logging
import model_transform as mt


seed = 42
torch.manual_seed(seed)

dict={}
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
#parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--using-amp_loss', '-amp_loss', dest='amp_loss', action='store_true',
                    help='using amp_loss mix f16 and f32')
parser.add_argument('--s', type=float, default=1e-5,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.010)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--quant', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be quant')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')
parser.add_argument('--warm_up_epochs', default=5, type=int,
                    help='warm_up_epochs of the neural network')
parser.add_argument('--save', default='./logs/',
                    help='checkpoint__save')
parser.add_argument('--filename', default='',
                    help='filename__save')
parser.add_argument('--quantization_bits', default=8, type=int,
                    help='quan_bits')
parser.add_argument('--m_bits', default=12, type=int,
                    help='m_bits')
parser.add_argument('--bias_bits', default=16, type=int,
                    help='acc_bits')
parser.add_argument('-M', '--Mn_aware', action='store_true',
                        help="Mn_aware", default=False)
parser.add_argument('--save_quantized_layer', action='store_true', help='if true, save the quantized layer')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

with open('./train_list_test.txt', 'r') as f:
    lines = f.readlines()
    lines = [i.strip('\n') for i in lines]

root_dir = lines

args.multi_patch = True
args.img_size = 288
args.color_mode = 'YUV_bt601V'

if args.Mn_aware == False:
    print("loading.....")
    args.quant = args.save + args.quant + '.pth'
    checkpoint = torch.load(args.quant)
    args.refine = args.save + args.refine + '.pth'
    checkpoint1 = torch.load(args.refine)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
    model = ClassifyNet(cfg=checkpoint1['cfg'])
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model = tflite_quantization_PACT_weight_and_act.replace(model=model,
                                                            quantization_bits=args.quantization_bits,
                                                            m_bits=args.m_bits,
                                                            bias_bits=args.bias_bits,
                                                            inference_type="all_fp",
                                                            Mn_aware=args.Mn_aware)

else:
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
    args.quant = args.save + args.quant + '.pth'
    checkpoint = torch.load(args.quant)
    args.refine = args.save + args.refine + '.pth'
    checkpoint1 = torch.load(args.refine)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet

    model = ClassifyNet(cfg=checkpoint1['cfg'])
    model = tflite_quantization_PACT_weight_and_act.replace(model=model,
                                                            quantization_bits=args.quantization_bits,
                                                            m_bits=args.m_bits,
                                                            bias_bits=args.bias_bits,
                                                            inference_type="full_int",
                                                            Mn_aware=args.Mn_aware)
    print(checkpoint.keys())
    if 'best_prec1' in checkpoint.keys():
        print(checkpoint['best_prec1'])
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

if args.cuda:
    model.cuda()

model, _ = tflite_quantization_PACT_weight_and_act.layer_transform(model)


model.cuda()

device = "cuda:0"


def get_train_label_list():
    path = './small_sign_classify_class_list.txt'
    labellist = list()
    with open(path, 'r') as f:
        for line in f:
            i = line.strip('\n')
            labellist.append(i)
    return labellist

def transform(img):

    img = img.transpose(2, 0, 1)
    img = img.astype(float) / 256.0

    return img


def test(model, img):
    model.eval()
    test_loss = 0
    correct = 0
    img_lst = []
    fail_lst = []
    labels = {}
    img_lst.append(img)
    file_lst = []
    labellist = get_train_label_list()
    for img_path in img_lst:
        temp_label = img_path.split('/')
        try:
            labels[img_path] = labellist.index(temp_label[-2])
        except:
            try:
                labels[img_path] = labellist.index(temp_label[-3])
            except:
                try:
                    labels[img_path] = labellist.index(temp_label[-4])
                except:
                    try:
                        labels[img_path] = labellist.index(temp_label[-5])
                    except:
                        fail_lst.append(img_path)
                        continue
        file_lst.append(img_path)

    img = np.zeros([288, 288, 3])
    label = np.zeros([3, 3], dtype=np.float32)

    ## 这里相当于一拖8，做一个随机拼接图像的操作，完成随机输入
    for i in range(9):
        img_path = file_lst[0]
        temp_img = cv2.imread(img_path)
        temp_img = cv2.resize(temp_img, (96, 96))

        if i == 0:  # 放中间
            img[96 * 1:96 * 1 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
            label[1, 1] = labels[img_path]
        elif i == 1:  # 一号位
            img[96 * 0:96 * 0 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
            label[0, 0] = labels[img_path]
        elif i == 2:  # 二号位
            img[96 * 0:96 * 0 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
            label[0, 1] = labels[img_path]
        elif i == 3:  #
            img[96 * 0:96 * 0 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
            label[0, 2] = labels[img_path]
        elif i == 4:
            img[96 * 1:96 * 1 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
            label[1, 0] = labels[img_path]
        elif i == 5:
            img[96 * 1:96 * 1 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
            label[1, 2] = labels[img_path]
        elif i == 6:
            img[96 * 2:96 * 2 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
            label[2, 0] = labels[img_path]
        elif i == 7:
            img[96 * 2:96 * 2 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
            label[2, 1] = labels[img_path]
        elif i == 8:
            img[96 * 2:96 * 2 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
            label[2, 2] = labels[img_path]

    img = transform(img)
    img = torch.from_numpy(img).float().cuda()
    label = torch.from_numpy(label).float().cuda()

    data = img.unsqueeze(0)
    output = model(data)

    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    return img, label, pred, correct

path = '/mnt/cephfs/home/lyy/data/tsr/t1q_tsr_data_part2/pl10/0###SZ_1_1_17_20170613_093948_3ae8f77b.pickle_04700_655_190_698_233.jpg_003030_YUV_bt601V.jpg'
img, label, pred, correct = test(model, path)
print("img", img)
print("label", label)
print("pred", pred)
print("correct", correct)











