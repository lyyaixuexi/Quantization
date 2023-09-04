import timm
import torch
import thop
import importlib
import os, sys
import numpy as np
# model_list = timm.list_models()
# print(model_list)
optimal_batch_size = 1
gpu = 0
path = "/mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/tsr_after_prune_0.1_part2.pth"
checkpoint = torch.load(path)
from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
print(checkpoint.keys())
model = ClassifyNet(cfg=checkpoint['cfg'])
# model = ClassifyNet()
# print(checkpoint['best_prec1'])
model.eval()
model.cuda(gpu)

torch.cuda.empty_cache()
print("batch_size: ", optimal_batch_size)
dummy_input = torch.randn(optimal_batch_size, 3, 288, 288, dtype=torch.float).to(gpu)
FLOPs, params = thop.profile(model, inputs=(dummy_input, ))
GFLOPs = FLOPs / 1E9 * 2
print("GFLOPs", GFLOPs)
print("params", params)
# repetitions = 100
# total_time = 0
# with torch.no_grad():
#   for rep in range(repetitions):
#      starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
#      starter.record()
#      _ = model(dummy_input)
#      ender.record()
#      torch.cuda.synchronize()
#      curr_time = starter.elapsed_time(ender)/1000
#      total_time += curr_time
# Throughput = (repetitions*optimal_batch_size)/total_time
# print('Final Throughput:',Throughput)




