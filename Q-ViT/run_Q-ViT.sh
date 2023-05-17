GPU_ID=2,3,4,5,6,7

#Train Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12223 --nproc_per_node=6 --use_env main.py \
        --model fourbits_deit_tiny_patch16_224 \
        --epochs 300 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 84 \
        --data-path /mnt/cephfs/mixed/dataset/imagenet \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./dist_4bit_tiny_lamb_3e-4_300_84 \
        --distillation-type hard \
        --teacher-model deit_tiny_patch16_224 \
        --opt fusedlamb \
        --resume /mnt/cephfs/home/lyy/Quantization/Q-ViT/dist_4bit_tiny_lamb_3e-4_300_84/checkpoint.pth \
        --start_epoch 252