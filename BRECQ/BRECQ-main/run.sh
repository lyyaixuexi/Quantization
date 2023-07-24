wget https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar
mv resnet18_imagenet.pth.tar ~/.cache/torch/checkpoints


GPU_ID=3,4,5
CUDA_VISIBLE_DEVICES=3,4,5 python main_imagenet.py --data_path /mnt/cephfs/mixed/dataset/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration