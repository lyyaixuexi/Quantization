GPU_ID=1,2

#eval Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12222 --nproc_per_node=2 --use_env main.py \
        --model fourbits_deit_tiny_patch16_224 \
        --epochs 300 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 84 \
        --data-path /mnt/cephfs/mixed/dataset/imagenet \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./test \
        --distillation-type hard \
        --teacher-model deit_tiny_patch16_224 \
        --opt fusedlamb \
        --eval