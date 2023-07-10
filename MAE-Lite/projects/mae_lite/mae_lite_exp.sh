GPU_ID=4,5,6,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e

ssl_train -b 1024 -d 0-4 -e 1000 -f finetuning_rpe_exp.py --amp \
--ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/outputs/mae_lite/mae_tiny_400e_numheads6/last_epoch_ckpt.pth.tar \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e_numheads6