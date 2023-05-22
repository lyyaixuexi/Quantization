GPU_ID=1,5,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --batch-size 128 --devices 4,5,6,7 -f /mnt/cephfs/home/lyy/Quantization/MAE-Lite/projects/mae_lite/mae_lite_exp.py --exp-options exp_name=mae_lite/mae_tiny_400e