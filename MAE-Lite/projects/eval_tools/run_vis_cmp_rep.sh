GPU_ID=4,5,6,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python run_vis_cmp_rep.py -b 64 --device cuda \
--save_name1 "MAE-Tiny" --ckpt1 "/mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e_ft_rpe_1000e.pth.tar" \
--exp-options1 pretrain_exp_name=mae_lite/mae_tiny_400e \
--save_name2 "DeiT-Tiny" --ckpt2 "/mnt/cephfs/home/lyy/Quantization/MAE-Lite/model/mae_tiny_400e_ft_rpe_1000e.pth.tar" \
--exp-options2 pretrain_exp_name=scratch/deit_tiny_300e