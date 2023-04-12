GPU_ID=4,5,6,7

#Train Q-ViT Deit-S 2/3/4bits:
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port=12223 --nproc_per_node=4 --use_env main.py \
        --model fourbits_deit_base_patch16_224 \
        --epochs 60 \
        --warmup-epochs 0 \
        --weight-decay 0. \
        --batch-size 2 \
        --data-path /mnt/cephfs/mixed/dataset/imagenet \
        --lr 3e-4 \
        --repeated-aug \
        --output_dir ./dist_4bit_base_lamb_3e-4_300_path8 \
        --distillation-type hard \
        --teacher-model deit_base_patch16_224 \
        --opt fusedlamb