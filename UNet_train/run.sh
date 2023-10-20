CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 U-net_train_val_mul.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node 2 U-net_train_val_mul.py
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node 2 main.py