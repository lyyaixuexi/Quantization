GPU_ID=4,5,6,7


CUDA_VISIBLE_DEVICES=${GPU_ID} python mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e