import torch
import random
# x=[]
# lr=0.010
# for epoch in range(0,100):
#     x.append(lr*((epoch+1) / 5) if (epoch+1) <= 5 else lr*0.5 * (
#                     math.cos((epoch - 5) / (101 - 5) * math.pi) + 1))
# print(x)
# m=[x for x in range(100)]
# t=x[-1]
# print(t)
# plt.plot(m,x)
# plt.show()
checkpoint = torch.load("/mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/pruned_tsr_0.1_gpu010_part2_best.pth")
print(checkpoint.keys())
print(checkpoint['best_prec1'])
print(checkpoint['epoch'])
# file_lst = [x for x in range(10000)]
# print(file_lst)
# random.seed(42)
# random.shuffle(file_lst)
# print(file_lst)
