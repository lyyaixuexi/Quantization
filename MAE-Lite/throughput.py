import timm
import torch
import numpy as np
# model_list = timm.list_models()
# print(model_list)

model_name = 'deit_tiny_patch16_224'

model = timm.create_model(model_name, pretrained=True)
print("model: ", model_name)
torch.cuda.set_device(1)
model.cuda(1)
# dummy_input = torch.randn(1, 3, 224, 224,dtype=torch.float).to(1)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#    _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#   for rep in range(repetitions):
#      starter.record()
#      _ = model(dummy_input)
#      ender.record()
#      # WAIT FOR GPU SYNC
#      torch.cuda.synchronize()
#      curr_time = starter.elapsed_time(ender)
#      timings[rep] = curr_time
# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# mean_fps = 1000. / mean_syn
# print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
# print(mean_syn)

optimal_batch_size = 2048
print("batch_size: ", optimal_batch_size)
dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(1)
repetitions=100
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
