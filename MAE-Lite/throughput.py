import timm
import torch
import importlib
import os, sys
import numpy as np
# model_list = timm.list_models()
# print(model_list)
optimal_batch_size = 2500
gpu = 0
model_name = 'vit_base_patch16_384.augreg_in1k'

# model = timm.create_model(model_name, pretrained=True)
# print("model: ", model_name)
# torch.cuda.set_device(gpu)
# model.cuda(gpu)


exp_file = 'projects/eval_tools/finetuning_rpe_exp.py'
# current_exp_name = os.path.basename(exp_file).split(".")[0]
sys.path.insert(0, os.path.dirname(exp_file))
current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
exp = current_exp.Exp(optimal_batch_size)
model = exp.get_model()
ckpt = torch.load('/root/mae_tiny_400e_ft_rpe_1000e.pth.tar', map_location="cpu")
msg = model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt["model"].items()})
model.eval()
model.cuda(gpu)

print("batch_size: ", optimal_batch_size)
dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(gpu)
repetitions = 100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*optimal_batch_size)/total_time
print('Final Throughput:',Throughput)
