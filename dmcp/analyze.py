import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
import argparse
from model_analyse import ModelAnalyse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg_0.4.txt', help='cfg for the pruned net')
parser.add_argument('--model_path', type=str, default='logs/ep058.pth')
parser.add_argument('--prune', action='store_true',  default=False)

# quantization
parser.add_argument('--inference_type', type=str, default='all_fp') #####all_fp/full_int
parser.add_argument("--quantization", action='store_true',  default=False)
parser.add_argument('--quantization_bits', type=int, default=6)
parser.add_argument('--m_bits', type=int, default=12)
parser.add_argument('--bias_bits', type=int, default=16) 
parser.add_argument('--fuseBN',  action='store_true',  default=False) 
parser.add_argument('--Mn_aware', action='store_true',  default=False)
parser.add_argument('--input_size', type=int, default=416)
if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    args=parser.parse_args()
    
    map_mode        = 0
    #-------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #-------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit' #'cocodata/xml/VOCdevkit'
    val_data_dir='VOCdevkit/VOC2007/JPEGImages'#'/gdata/MSCOCO2017/val2017'
    val_txt_file="VOC2007/ImageSets/Main/val.txt"#"VOC2007/ImageSets/Main/val2017.txt"
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    if args.quantization_bits>4 and args.quantization_bits<=8:
        bits=8
    elif args.quantization_bits>8:
        bits=16
    else:
         bits=4
    if args.inference_type=='full_int':
        map_out_path    = 'map_out_int{}'.format(bits)
    else:
        map_out_path    = 'map_out_float_{}_bits'.format(bits)
 
    class_names, _ = get_classes(classes_path)
     
    print('madel analyze,input_size=416.....................')
    print("Load model.")
    yolo = YOLO(confidence = 0.001, nms_iou = 0.5,model_path=args.model_path,yolo_tiny_cfg=args.cfg,prune=args.prune, inference_type=args.inference_type,quantization=args.quantization,quantization_bits=args.quantization_bits,m_bits=args.m_bits,bias_bits=args.bias_bits,fuseBN=args.fuseBN, Mn_aware=args.Mn_aware)
    
    model_analyse = ModelAnalyse(yolo.net)
    model_analyse.params_count()
    model_analyse.flops_compute(torch.randn(1, 3, args.input_size,  args.input_size)) 