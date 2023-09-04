from models.experimental import attempt_load_pruned
import torch
import sys
sys.path.append('/home/wenzhiquan/wangliqin/Code/new_try_yolov5_unet/yolov5_UNet/UNet')
sys.path.append('/home/wenzhiquan/wangliqin/Code/new_try_yolov5_unet/yolov5_UNet/UNet/unet')
sys.path.append('/home/wenzhiquan/wangliqin/Code/new_try_yolov5_unet/yolov5_UNet/UNet/utils')

# dict_keys(['epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'wandb_id', 'date'])

# weights='finetune_prune.pt'
weights='weight_finetune/fp32.pt'
pt_file=torch.load(weights)
# print(pt_file['best_fitness'])
pt_file['best_fitness']=None
pt_file['ema']=None
pt_file['updates']=None
pt_file['optimizer']=None
pt_file['wandb_id']=None
torch.save(pt_file, 'save.pt')


# model = pt_file['model']
# print(model)
# torch.save(model.state_dict(), 'save.pt')
# for name, p in model.named_parameters():
#         # print(name)
#     print(p)
#     break
    # print(...)
# print(model.dtype)