GPU_ID=0,1,2,3,4,5,6,7

#Train Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12222 --nproc_per_node=8 --use_env main.py \
        --model fourbits_deit_tiny_patch16_224 \
        --epochs 60 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 64 \
        --data-path /mnt/cephfs/mixed/dataset/imagenet \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./dist_4bit_tiny_lamb_3e-4_300_512 \
        --distillation-type hard \
        --teacher-model deit_tiny_patch16_224 \
        --opt fusedlamb