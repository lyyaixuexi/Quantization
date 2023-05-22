GPU_ID=4,5,6,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --devices 4,5,6,7 --batch-size 128 -f /mnt/cephfs/home/lyy/Quantization/MAE-Lite/projects/mae_lite/mae_lite_exp.py --exp-options exp_name=mae_lite/mae_tiny_400e --ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e.pth.tar