import torch
import os
import sys

sys.path.append(sys.path[0]+'/UNet')
sys.path.append(sys.path[0]+'/UNet/unet')
sys.path.append(sys.path[0]+'/UNet/utils')

checkpoint = torch.load("/mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w4a4b16/weights/best.pt")

print(checkpoint.keys())
print(checkpoint['model'].state_dict())