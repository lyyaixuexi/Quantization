GPU_ID=1,5,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --batch-size 128 --devices 4,5,6,7 -f /mnt/cephfs/home/lyy/Quantization/MAE-Lite/projects/mae_lite/mae_lite_exp.py --exp-options exp_name=mae_lite/mae_tiny_400e

ssl_train -b 1024 -d 0-7 -e 1000 -f finetuning_rpe_exp.py --amp \
--ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/outputs/mae_lite/mae_tiny_400e_numheads8/last_epoch_ckpt.pth.tar --exp-options pretrain_exp_name=mae_lite/mae_tiny_400e_numheads8

ssl_train -b 4096 -d 0 -e 400 -f mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e_numheads8