python -m torch.distributed.launch --nproc_per_node=4 --master_port=62235 --use_env main.py imagenet_resnet_clip.hocon 0
