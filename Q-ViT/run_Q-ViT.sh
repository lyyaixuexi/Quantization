GPU_ID=2,3,4,5

#Train Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12222 --nproc_per_node=2 --use_env main.py \
        --model fourbits_deit_small_patch16_224 \
        --epochs 300 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 32 \
        --data-path /mnt/cephfs/mixed/dataset/imagenet \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./dist_4bit_small_lamb_3e-4_300_512 \
        --distillation-type hard \
        --teacher-model deit_small_patch16_224 \
        --opt fusedlamb