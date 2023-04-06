GPU_ID=2,3,4

#Train Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12345 --nproc_per_node=3 --use_env main.py \
        --model fourbits_deit_small_patch16_224 \
        --epochs 300 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 128 \
        --data-path /mnt/ssd/datasets/imagenet/ \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./dist_4bit_small_lamb_3e-4_300_512 \
        --distillation-type hard \
        --teacher-model deit_small_patch16_224 \
        --opt fusedlamb