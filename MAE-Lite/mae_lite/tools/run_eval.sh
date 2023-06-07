GPU_ID=4,5,6,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --devices 4,5,6,7 --batch-size 128 -f /mnt/cephfs/home/lyy/Quantization/MAE-Lite/projects/mae_lite/mae_lite_exp.py --exp-options exp_name=mae_lite/mae_tiny_400e --ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e.pth.tar


ssl_eval -b 128 -d 0-7 -f mae_lite_exp.py \
--exp-options exp_name=mae_lite/mae_tiny_400e \
--ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e.pth.tar

ssl_train -b 128 -d 0-7 -e 400 -f mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e

python mae_lite/tools/eval.py -b 128 -d 0 -f projects/eval_tools/finetuning_rpe_exp.py \
--ckpt /mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e_ft_rpe_1000e.pth.tar \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e/ft_rpe_eval