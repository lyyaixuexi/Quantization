python -m torch.distributed.launch --nproc_per_node=4 --master_port=65525 --use_env main.py imagenet_resnet.hocon 0 0.05 16 0,1,2,3