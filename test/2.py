import timm
import torch
import importlib
import os, sys
import numpy as np
# model_list = timm.list_models()
# print(model_list)
optimal_batch_size = 512
gpu = 0
model_name = 'deit_tiny_patch16_224'
#
sys.path.insert(0, os.path.dirname(exp_file))
current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
exp = current_exp.Exp(optimal_batch_size)
model = exp.get_model()

ckpt = torch.load('../MAE-Lite/model/mae_tiny_400e_ft_300e.pth.tar', map_location="cpu")
ckpt["model"]['module.model.norm.weight'] = ckpt["model"].pop('module.model.fc_norm.weight')
ckpt["model"]['module.model.norm.bias'] = ckpt["model"].pop('module.model.fc_norm.bias')

msg = model.load_state_dict({k.replace('module.model.', ''): v for k, v in ckpt["model"].items()})
model.eval()
model.cuda(gpu)


print("batch_size: ", optimal_batch_size)
dummy_input = torch.ones(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(gpu)
repetitions = 100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*optimal_batch_size)/total_time
print('Final Throughput:',Throughput)






