import torch


checkpoint1 = torch.load("/mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/tsr_after_prune_0.1.pth")
print(checkpoint1['state_dict'].keys())
checkpoint2 = torch.load("/mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/pruned_tsr_0.1.pth")
print(checkpoint2['state_dict'].keys())
print(checkpoint1['state_dict'].keys() == checkpoint2['state_dict'].keys())

