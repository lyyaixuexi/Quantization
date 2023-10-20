# 云脑容器
dockerhub.pcl.ac.cn:5000/user-images/lihk:pt18_deforable_detr_timm

# 云脑运行指令
bash
cd /userhome/UNet && xxxx

############################################ train fp32 model #################################################################
# fp32   50 epoch   lr0.0001   mpa: 88.45%   mIoU: 0.5497
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -e 50 -b 2 -c ./checkpoint/ckpt_fp32_50epoch_20231012

# fp32  100 epoch   lr0.0001   mpa: 88.67%  mIoU: 0.5510
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -e 100 -b 2 -c ./checkpoint/ckpt_fp32_100epoch_20210901

# fp32  50 epoch  lr0.00001   best_43epoch: mpa: 88.37%  mIoU: 0.5524
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -c ./checkpoint/ckpt_fp32_50epoch_20210904_lr00001

# fp32  50 epoch  lr0.0001 best_48epoch: mpa: 88.38%  mIoU: 0.5572
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 50 -b 2 -c ./checkpoint/ckpt_fp32_50epoch_20210902

# fp32  50 epoch  lr0.001   best_49epoch: mpa: 88.30%  mIoU: 0.5583
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.001 -e 50 -b 2 -c ./checkpoint/ckpt_fp32_50epoch_20210904_lr001

# fp32  50 epoch  lr0.01    best_49epoch: mpa: 84.85%  mIoU: 0.4050
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.01 -e 50 -b 2 -c ./checkpoint/ckpt_fp32_50epoch_20210904

# fp32 200 epoch lr0.0001   best_160epoch mpa: 89.28%  mIoU: 0.5912   best result!
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 200 -b 4 -c ./checkpoint/ckpt_fp32_200epoch_20210914

# fp32  50 epoch  lr0.0001  best_47epoch  mpa: 88.45%  mIoU: 0.5484
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 50 -b 4 -c ./checkpoint/ckpt_fp32_50epoch_20210928

# fp32  50 epoch  lr0.0001  best_44epoch mpa: 87.70%  mIoU: 0.5219
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 50 -b 4 --half_channel -c ./checkpoint/ckpt_fp32_200epoch_half_channel_20210930

# fp32  200 epoch  lr0.0001  best_149epoch mpa: 88.75%  mIoU: 0.5683
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 200 -b 4 --half_channel -c ./checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200

# fp32  200 epoch  lr0.0001  best_158epoch mpa: 88.57%  mIoU: 0.5728
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 200 -b 4 --half_channel --strict_cin_number -c ./checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013.txt

# fp32  200 epoch  lr0.0001  survive only quarter channel   best_159epoch mpa: 88.03%  mIoU: 0.5404
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 200 -b 4 --quarter_channel --strict_cin_number -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102.txt

# fp32  200 epoch  lr0.0001  survive only quarter channel   best_187epoch mpa: 88.12%  mIoU: 0.5502
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py -l 0.0001 -e 200 -b 8 --quarter_channel --strict_cin_number -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu.txt


########################################### train quantize model stage1 #########################################################
# int16  100 epoch  lr0.0001   mpa: 89.37%  mIoU: 0.5949
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 16 -y 16 -z 32 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth -c ./checkpoint/ckpt_int16_30epoch_20210921_lr0001

# int8  100 epoch  lr0.0001    mpa: 89.24%  mIoU: 0.5904
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 8 -y 8 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth -c ./checkpoint/ckpt_int8_30epoch_20210921_lr0001

# int4  100 epoch  lr0.0001   mpa: 89.32%  mIoU: 0.5899
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 4 -y 4 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth -c ./checkpoint/ckpt_int4_30epoch_20210921_lr0001

#注意：stage1的超参y设置错误, 训练stage1的时候, 没弄清楚具体含义，没有正确设置y(x=>quantization_bits  y=>m_bits  z=>bias_bits)

# int16  100 epoch  lr0.0001   mpa: 89.48%  mIoU: 0.5929
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 16 -y 12 -z 32 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_50epoch_20210928/CP_best_epoch47.pth -c ./checkpoint/ckpt_int16_30epoch_20210929_lr0001

# int8  100 epoch  lr0.0001    mpa: 89.43%  mIoU: 0.5901
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 8 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_50epoch_20210928/CP_best_epoch47.pth -c ./checkpoint/ckpt_int8_30epoch_20210929_lr0001

# int4  100 epoch  lr0.0001    mpa: 89.42%  mIoU: 0.5896
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 -q -x 4 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_50epoch_20210928/CP_best_epoch47.pth -c ./checkpoint/ckpt_int4_30epoch_20210929_lr0001


# int16  100 epoch  lr0.0001   mpa: 88.67%  mIoU: 0.5689
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel -q -x 16 -y 12 -z 32 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200/CP_best_epoch149.pth -c ./checkpoint/ckpt_int16_100epoch_20211009_lr0001

# int8  100 epoch  lr0.0001    mpa: 88.83%  mIoU: 0.5737
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel -q -x 8 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200/CP_best_epoch149.pth -c ./checkpoint/ckpt_int8_100epoch_20211009_lr0001

# int4  100 epoch  lr0.0001    mpa: 87.98%  mIoU: 0.5631
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel -q -x 4 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200/CP_best_epoch149.pth -c ./checkpoint/ckpt_int4_100epoch_20211009_lr0001


# int16  100 epoch  lr0.0001   mpa: 88.91%  mIoU: 0.5773
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013/CP_best_epoch158.pth -c ./checkpoint/ckpt_int16_100epoch_20211014_lr0001 2>&1 | tee ./checkpoint/ckpt_int16_100epoch_20211014_lr0001.txt

# int8  100 epoch  lr0.0001    mpa: 88.68%  mIoU: 0.5697
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013/CP_best_epoch158.pth -c ./checkpoint/ckpt_int8_100epoch_20211014_lr0001 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20211014_lr0001.txt

# int4  100 epoch  lr0.0001    mpa: 88.73%  mIoU: 0.5684
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.0001 -e 100 -b 2 --half_channel --strict_cin_number -q -x 4 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013/CP_best_epoch158.pth -c ./checkpoint/ckpt_int4_100epoch_20211014_lr0001 2>&1 | tee ./checkpoint/ckpt_int4_100epoch_20211014_lr0001.txt


# int16  100 epoch  lr0.0001   初始化 mpa: 88.08% mIoU: 0.5484    最佳模型 mpa: 88.09%  mIoU: 0.5514
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int16_100epoch_20211103_lr0001 2>&1 | tee ./checkpoint/ckpt_int16_100epoch_20211103_lr0001.txt

# int8  100 epoch  lr0.0001  初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.26%  mIoU: 0.5510
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20211103_lr0001 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20211103_lr0001.txt

# int4  100 epoch  lr0.0001    mpa: 87.96%  mIoU: 0.5354
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 4 -y 12 -z 16 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int4_100epoch_20211103_lr0001 2>&1 | tee ./checkpoint/ckpt_int4_100epoch_20211103_lr0001.txt

### ablation study for overflow-aware
# int8  100 epoch  lr0.0001  without overflow-aware
# 初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.23%  mIoU: 0.5521
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp0 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp0.txt
# 初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.27%  mIoU: 0.5501
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp1 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp1.txt
# 初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.62%  mIoU: 0.5563
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2.txt


### different bit-width for accumulators
# int8  100 epoch  lr0.0001  初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.25%  mIoU:0.5488
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 20 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator20 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator20.txt
# int8  100 epoch  lr0.0001  初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.26%  mIoU:0.5494
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 18 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator18 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator18.txt
# int8  100 epoch  lr0.0001  初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.10%  mIoU:0.5437
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 14 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator14 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator14.txt
# int8  100 epoch  lr0.0001  初始化 mpa: 81.02% mIoU: 0.4282    最佳模型 mpa: 88.14%  mIoU:0.5477
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 12 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator12 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator12.txt
# int8  100 epoch  lr0.0001  初始化 mpa: 81.01% mIoU: 0.4281    最佳模型 mpa: 88.08%  mIoU:0.5463
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 10 --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator10 2>&1 | tee ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator10.txt


########################################### train quantize model stage2 #########################################################
# int16  50 epoch  lr0.00001  mpa: 89.68%  mIoU: 0.6013
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_30epoch_20210921_lr0001/CP_best_epoch75.pth -c ./checkpoint/ckpt_int16_50epoch_20210922_stage2

# int8  50 epoch  lr0.00001   mpa: 89.51%  mIoU: 0.5947
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_30epoch_20210921_lr0001/CP_best_epoch34.pth -c ./checkpoint/ckpt_int8_50epoch_20210922_stage2

# int4  50 epoch  lr0.00001   mpa: 89.71%  mIoU: 0.6000
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 4 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int4_30epoch_20210921_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int4_50epoch_20210922_stage2

# int16  50 epoch  lr0.00001  mpa: 89.66%  mIoU: 0.5996
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_30epoch_20210929_lr0001/CP_best_epoch87.pth -c ./checkpoint/ckpt_int16_50epoch_20210930_stage2

# int8  50 epoch  lr0.00001   mpa: 89.68%  mIoU: 0.5984
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_30epoch_20210929_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20210930_stage2

# int4  50 epoch  lr0.00001   mpa: 89.63%  mIoU: 0.5998
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 -q -x 4 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int4_30epoch_20210929_lr0001/CP_best_epoch73.pth -c ./checkpoint/ckpt_int4_50epoch_20210930_stage2

# int16  50 epoch  lr0.00001  mpa: 89.05  mIoU: 0.5816
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211009_lr0001/CP_best_epoch44.pth -c ./checkpoint/ckpt_int16_50epoch_20211010_stage2

# int8  50 epoch  lr0.00001   mpa: 89.00 mIoU: 0.5830
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211009_lr0001/CP_best_epoch72.pth -c ./checkpoint/ckpt_int8_50epoch_20211010_stage2

# int4  50 epoch  lr0.00001   mpa: 88.84  mIoU: 0.5789
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel -q -x 4 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int4_100epoch_20211009_lr0001/CP_best_epoch52.pth -c ./checkpoint/ckpt_int4_50epoch_20211010_stage2

# int16  50 epoch  lr0.00001  mpa: 89.23  mIoU: 0.5817
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211014_lr0001/CP_best_epoch72.pth -c ./checkpoint/ckpt_int16_50epoch_20211019_stage2 2>&1 | tee ./checkpoint/ckpt_int16_50epoch_20211019_stage2.txt

# int8  50 epoch  lr0.00001   mpa: 89.02  mIoU: 0.5788
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211014_lr0001/CP_best_epoch48.pth -c ./checkpoint/ckpt_int8_50epoch_20211019_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20211019_stage2.txt

# int4  50 epoch  lr0.00001   mpa: 89.04  mIoU: 0.5835
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py -l 0.00001 -e 50 -b 2 --half_channel --strict_cin_number  -q -x 4 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int4_100epoch_20211014_lr0001/CP_best_epoch77.pth -c ./checkpoint/ckpt_int4_50epoch_20211019_stage2 2>&1 | tee ./checkpoint/ckpt_int4_50epoch_20211019_stage2.txt

# int16  50 epoch  lr0.00001  初始化 mpa:61.73  mIou:0.2677   最佳模型 mpa: 88.53  mIoU: 0.5593
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211103_lr0001/CP_best_epoch28.pth -c ./checkpoint/ckpt_int16_50epoch_20211104_stage2 2>&1 | tee ./checkpoint/ckpt_int16_50epoch_20211104_stage2.txt
# 最佳模型 mpa: 88.57  mIoU: 0.5589
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211103_lr0001/CP_best_epoch28.pth -c ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp1 2>&1 | tee ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp1.txt
# 最佳模型 mpa: 88.49  mIoU: 0.5587
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211103_lr0001/CP_best_epoch28.pth -c ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp2.txt
# 最佳模型 mpa: 88.53  mIoU: 0.5604     best
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 16 -y 12 -z 32 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_100epoch_20211103_lr0001/CP_best_epoch28.pth -c ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp3 2>&1 | tee ./checkpoint/ckpt_int16_50epoch_20211104_stage2_exp3.txt


# int8  50 epoch  lr0.00001   初始化 mpa:64.41 mIou:0.2934   最佳模型 mpa: 88.53  mIoU: 0.5593  best
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20211104_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20211104_stage2.txt
# 最佳模型 mpa: 88.50  mIoU: 0.5590
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp1 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp1.txt
# 最佳模型 mpa: 88.56  mIoU: 0.5588
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp2.txt
# 最佳模型 mpa: 88.54  mIoU: 0.5580   worst
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp3 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20211104_stage2_exp3.txt


# int4  50 epoch  lr0.00001   mpa: 88.26  mIoU: 0.5489
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number  -q -x 4 -y 12 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int4_100epoch_20211103_lr0001/CP_best_epoch78.pth -c ./checkpoint/ckpt_int4_50epoch_20211104_stage2 2>&1 | tee ./checkpoint/ckpt_int4_50epoch_20211104_stage2.txt

### ablation study for overflow-aware
# int8  50 epoch  lr0.00001   without overflow-aware
# 初始化 mpa:60.61 mIou:0.2121   最佳模型 mpa: 88.37  mIoU: 0.5469
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2/CP_best_epoch93.pth -c ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp0 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp0.txt
# 初始化 mpa:60.61 mIou:0.2121   最佳模型 mpa: 88.40  mIoU: 0.5456
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2/CP_best_epoch93.pth -c ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp1 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp1.txt
# 初始化 mpa:60.61 mIou:0.2121   最佳模型 mpa: 88.41  mIoU: 0.5467
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2/CP_best_epoch93.pth -c ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp2.txt
# 初始化 mpa:60.61 mIou:0.2121   最佳模型 mpa: 88.41  mIoU: 0.5467
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 16 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_without_OWQ_20220225_lr0001_exp2/CP_best_epoch93.pth -c ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp2.txt


### different bit-width for accumulators
# int8  50 epoch  lr0.00001
# 初始化 mpa:69.87 mIou:0.3105   最佳模型 mpa: 88.56  mIoU: 0.5554
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 20 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator20/CP_best_epoch77.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_accumulator20_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_accumulator20_stage2.txt
# 初始化 mpa:67.46 mIou:0.2878   最佳模型 mpa: 88.59  mIoU: 0.5553
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 18 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator18/CP_best_epoch77.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_accumulator18_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_accumulator18_stage2.txt
# 初始化 mpa:68.39 mIou:0.3152   最佳模型 mpa: 88.46  mIoU: 0.5542
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 14 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator14/CP_best_epoch89.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_accumulator14_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_accumulator14_stage2.txt
# 初始化 mpa:67.14 mIou:0.3109   最佳模型 mpa: 88.11  mIoU: 0.5463
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 12 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator12/CP_best_epoch34.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_accumulator12_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_accumulator12_stage2.txt
# 初始化 mpa:27.55 mIou:0.0386   最佳模型 mpa: 12.48 mIoU: 0.0062
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q -x 8 -y 12 -z 10 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20220227_lr0001_accumulator10/CP_best_epoch77.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_accumulator10_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_accumulator10_stage2.txt

### different bit-width for multipliers
# multipliers=2  初始化 mpa:12.48 mIou:0.0062   最佳模型 mpa: 16.04 mIoU: 0.0362
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 2 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M2_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M2_stage2.txt
# multipliers=3  初始化 mpa:25.18 mIou:0.0324   最佳模型 mpa: 87.78 mIoU: 0.5300
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 3 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M3_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M3_stage2.txt
# multipliers=4  初始化 mpa:50.73 mIou:0.1828   最佳模型 mpa: 88.39 mIoU: 0.5543
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 4 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M4_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M4_stage2.txt
# multipliers=8  初始化 mpa:64.35 mIou:0.2914   最佳模型 mpa: 88.49 mIoU: 0.5589
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 8 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2.txt
# multipliers=8  初始化 mpa:64.35 mIou:0.2914   最佳模型 mpa: 88.53 mIoU: 0.5582
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 8 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2_exp1 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2_exp1.txt
# multipliers=8  初始化 mpa:64.35 mIou:0.2914   最佳模型 mpa: 88.53 mIoU: 0.5578
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 8 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M8_stage2_exp2.txt
# multipliers=18  初始化 mpa:64.43 mIou:0.2937   最佳模型 mpa: 88.58 mIoU: 0.5595
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 18 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2.txt
# multipliers=18  初始化 mpa:64.43 mIou:0.2937   最佳模型 mpa: 88.52 mIoU: 0.5588
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 18 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2_exp1 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2_exp1.txt
# multipliers=18  初始化 mpa:64.43 mIou:0.2937   最佳模型 mpa: 88.55 mIoU: 0.5591
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 18 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2_exp2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M18_stage2_exp2.txt
# multipliers=24  初始化 mpa:64.43 mIou:0.2942   最佳模型 mpa: 88.52 mIoU: 0.5576
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 24 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M24_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M24_stage2.txt
# multipliers=32  初始化 mpa:64.43 mIou:0.2943   最佳模型 mpa: 88.48 mIoU: 0.5584
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py -l 0.00001 -e 50 -b 8 --quarter_channel --reset_M --strict_cin_number -q -x 8 -y 32 -z 16 --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int8_100epoch_20211103_lr0001/CP_best_epoch97.pth -c ./checkpoint/ckpt_int8_50epoch_20220227_reset_M32_stage2 2>&1 | tee ./checkpoint/ckpt_int8_50epoch_20220227_reset_M32_stage2.txt

########################################## evaluate fp32 model##############################################################
# mpa: 89.56%  mIoU: 0.5943
python eval.py -b 2 -m model_best/checkpoint_fp32/CP_epoch26.pth

# mpa: 88.45%   mIoU: 0.5497
python eval.py -b 2 -m checkpoint/ckpt_fp32_50epoch_20210901/CP_epoch50.pth

# mpa: 88.67%  mIoU: 0.5510
python eval.py -b 2 -m checkpoint/ckpt_fp32_100epoch_20210901/CP_epoch100.pth

# mpa: 88.38%  mIoU: 0.5572
python eval.py -b 2 -m checkpoint/ckpt_fp32_50epoch_20210902/CP_best_epoch48.pth

# mpa: 89.28%  mIoU: 0.5912
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth

# mpa: 88.45%  mIoU: 0.5484
python eval.py -b 4 -m checkpoint/ckpt_fp32_50epoch_20210928/CP_best_epoch47.pth

# mpa: 87.70%  mIoU: 0.5219
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_half_channel_20210930/CP_best_epoch44.pth --half_channel

# mpa: 88.55%  mIoU: 0.5537
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200/CP_best_epoch93.pth --half_channel

# mpa: 88.75%  mIoU: 0.5683
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_half_channel_20210930_real_200/CP_best_epoch149.pth --half_channel

# mpa: 88.57%  mIoU: 0.5728
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_half_channel_strict_cin_number_20211013/CP_best_epoch158.pth --half_channel --strict_cin_number

# mpa: 88.12%  mIoU: 0.5502
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth --quarter_channel --strict_cin_number 2>&1 | tee ./checkpoint/eval_ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu.txt


################################################### eval quantization stage1 ####################################################
# mpa: 89.37%  mIoU: 0.5949
python eval.py -b 2 -t q -x 16 -y 16 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_30epoch_20210921_lr0001/CP_best_epoch75.pth

# mpa: 89.24%  mIoU: 0.5904
python eval.py -b 2 -t q -x 8 -y 8 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_30epoch_20210921_lr0001/CP_best_epoch34.pth

# mpa: 89.32%  mIoU: 0.5899
python eval.py -b 2 -t q -x 4 -y 4 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_30epoch_20210921_lr0001/CP_best_epoch97.pth

# mpa: 89.48%  mIoU: 0.5929
python eval.py -b 2 -t q -x 16 -y 16 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_30epoch_20210929_lr0001/CP_best_epoch87.pth

# mpa: 89.43%  mIoU: 0.5901
python eval.py -b 2 -t q -x 8 -y 8 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_30epoch_20210929_lr0001/CP_best_epoch97.pth

# mpa: 89.42%  mIoU: 0.5896
python eval.py -b 2 -t q -x 4 -y 4 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_30epoch_20210929_lr0001/CP_best_epoch73.pth

# mpa: 88.67%  mIoU: 0.5689
python eval.py -b 2 -t q -x 16 -y 12 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_100epoch_20211009_lr0001/CP_best_epoch44.pth --half_channel

# mpa: 88.83%  mIoU: 0.5737
python eval.py -b 2 -t q -x 8 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_100epoch_20211009_lr0001/CP_best_epoch72.pth --half_channel

# mpa: 87.98%  mIoU: 0.5631
python eval.py -b 2 -t q -x 4 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_100epoch_20211009_lr0001/CP_best_epoch52.pth --half_channel



################################################### eval quantization stage2 ####################################################
# mpa: 89.68%  mIoU: 0.6013
python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_50epoch_20210922_stage2/CP_best_epoch6.pth
# mpa: 89.68%  mIoU: 0.6013
python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20210922_stage2/CP_best_epoch6.pth

# mpa: 89.51%  mIoU: 0.5947
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_50epoch_20210922_stage2/CP_best_epoch2.pth
# mpa: 89.51%  mIoU: 0.5947
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20210922_stage2/CP_best_epoch2.pth

# mpa: 89.71%  mIoU: 0.6000
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_50epoch_20210922_stage2/CP_best_epoch19.pth
# mpa: 89.71%  mIoU: 0.6000
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20210922_stage2/CP_best_epoch19.pth

# mpa: 89.66%  mIoU: 0.5996
python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_50epoch_20210930_stage2/CP_best_epoch9.pth
# mpa: 89.66%  mIoU: 0.5996
python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20210930_stage2/CP_best_epoch9.pth

# mpa: 89.68%  mIoU: 0.5984
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_50epoch_20210930_stage2/CP_best_epoch9.pth
# mpa: 89.68%  mIoU: 0.5984
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20210930_stage2/CP_best_epoch9.pth

# mpa: 89.63%  mIoU: 0.5998
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_50epoch_20210930_stage2/CP_best_epoch9.pth
# mpa: 89.63%  mIoU: 0.5998
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20210930_stage2/CP_best_epoch9.pth

# mpa: 88.84  mIoU: 0.5789
python eval.py -b 2 --half_channel -t qfm -x 4 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int4_50epoch_20211010_stage2/CP_best_epoch17.pth
# mpa: 88.84  mIoU: 0.5789
python eval.py -b 2 --half_channel -t qfm -x 4 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20211010_stage2/CP_best_epoch17.pth

# mpa: 89.00 mIoU: 0.5830
python eval.py -b 2 --half_channel -t qfm -x 8 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_50epoch_20211010_stage2/CP_best_epoch4.pth
# mpa: 89.00 mIoU: 0.5830
python eval.py -b 2 --half_channel -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211010_stage2/CP_best_epoch4.pth

# mpa: 89.05  mIoU: 0.5816
python eval.py -b 2 --half_channel -t qfm -x 16 -y 12 -z 32 --inference_type all_fp -m checkpoint/ckpt_int16_50epoch_20211010_stage2/CP_best_epoch17.pth
# mpa: 89.05  mIoU: 0.5816
python eval.py -b 2 --half_channel -t qfm -x 16 -y 12 -z 32 --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20211010_stage2/CP_best_epoch17.pth

##### test quarter_channel model
# mpa: 88.53  mIoU: 0.5604
python eval.py -b 8 --quarter_channel -t qfm -x 16 -y 12 -z 32 --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20211104_stage2_exp3/CP_best_epoch5.pth
# mpa: 88.54  mIoU: 5580
python eval.py -b 8 --quarter_channel -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211104_stage2_exp3/CP_best_epoch43.pth
# mpa: 88.26  mIoU: 0.5489
python eval.py -b 8 --quarter_channel -t qfm -x 4 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20211104_stage2/CP_best_epoch21.pth

# mpa: 88.37  mIoU: 0.5469
python eval.py -b 8 --quarter_channel -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_without_OWQ_20220225_stage2_exp0/CP_best_epoch46.pth


################################################## save integer model ###############################################################
python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20210922_stage2/CP_best_epoch6.pth --save_model --result_name data/Unet_output_int16_original
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20210922_stage2/CP_best_epoch2.pth --save_model --result_name data/Unet_output_int8_original
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20210922_stage2/CP_best_epoch19.pth --save_model --result_name data/Unet_output_int4_original

python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --half_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20211019_stage2/CP_best_epoch1.pth --save_model --result_name data/Unet_output_int16_half_channel_strict
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --half_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211019_stage2/CP_best_epoch8.pth --save_model --result_name data/Unet_output_int8_half_channel_strict
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --half_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20211019_stage2/CP_best_epoch26.pth --save_model --result_name data/Unet_output_int4_half_channel_strict

python eval.py -b 2 -t qfm -x 16 -y 12 -z 32 --quarter_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int16_50epoch_20211104_stage2_exp3/CP_best_epoch5.pth --save_model --result_name data/Unet_output_int16_quarter_channel_strict
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --quarter_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211104_stage2_exp3/CP_best_epoch43.pth --save_model --result_name data/Unet_output_int8_quarter_channel_strict
python eval.py -b 2 -t qfm -x 4 -y 12 -z 16 --quarter_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int4_50epoch_20211104_stage2/CP_best_epoch21.pth --save_model --result_name data/Unet_output_int4_quarter_channel_strict



############################# Load Integer model #########################################
python load_scale_and_param.py --scale_table_file data/Unet_output_int16_original.scale --quantized_param_file data/Unet_output_int16_original_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int8_original.scale --quantized_param_file data/Unet_output_int8_original_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int4_original.scale --quantized_param_file data/Unet_output_int4_original_param.npy

python load_scale_and_param.py --scale_table_file data/Unet_output_int16_half_channel_strict.scale --quantized_param_file data/Unet_output_int16_half_channel_strict_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int8_half_channel_strict.scale --quantized_param_file data/Unet_output_int8_half_channel_strict_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int4_half_channel_strict.scale --quantized_param_file data/Unet_output_int4_half_channel_strict_param.npy

python load_scale_and_param.py --scale_table_file data/Unet_output_int16_quarter_channel_strict.scale --quantized_param_file data/Unet_output_int16_quarter_channel_strict_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int8_quarter_channel_strict.scale --quantized_param_file data/Unet_output_int8_quarter_channel_strict_param.npy
python load_scale_and_param.py --scale_table_file data/Unet_output_int4_quarter_channel_strict.scale --quantized_param_file data/Unet_output_int4_quarter_channel_strict_param.npy



########################################################################################################################
################################### 16bit and 8 bit mixture quantization ###############################################
# install some python packages
# pip install pyhessian
# pip install pupl
# 保存为云脑容器
dockerhub.pcl.ac.cn:5000/user-images/lihk:mixture_unet

# 云脑运行指令
bash
cd /userhome/UNet && xxxx

########################################### train quantize model stage1 #########################################################
# int16&int8 mixture quantization   target_BOPS_rate: 0.5  100 epoch lr0.0001
# 初始化 mpa: 81.09% mIoU: 0.4291    最佳模型 mpa: 88.15%   mIoU: 0.5483
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp0 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp0.txt
# 初始化 mpa: 81.09% mIoU: 0.4291    最佳模型 mpa: 88.21%   mIoU: 0.5497
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp1 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp1.txt
# 初始化 mpa: 81.09% mIoU: 0.4291    最佳模型 mpa: 88.24%   mIoU: 0.5469
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp2 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp2.txt
# 初始化 mpa: 81.09% mIoU: 0.4291    最佳模型 mpa: 88.16%   mIoU: 0.5498
# 这是修复bug后，跑实验得到的最好的stage1模型，略低于纯int16的mIou（0.5514）和纯int8的mIou（0.5510），故使用这个模型初始化后续的stage2量化。
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.0001 -e 100 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -m ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20211102_2gpu/CP_best_epoch187.pth -c ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3.txt


########################################### train quantize model stage2 #########################################################
# int16&int8 mixture quantization   target_BOPS_rate: 0.5   50 epoch  lr0.00001
# 初始化 mpa: 64.09  mIoU: 0.2908  最佳模型 mpa:88.59   mIoU:0.5577
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp0 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp0.txt
# 初始化 mpa: 64.09  mIoU: 0.2908  最佳模型 mpa:88.57   mIoU:0.5568
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp3 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp3.txt
# 初始化 mpa: 64.09  mIoU: 0.2908  最佳模型 mpa:88.57   mIoU:0.5575
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp5 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp5.txt
# 最佳模型 mpa:88.57   mIoU:0.5574
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp6 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp6.txt
# 最佳模型 mpa:88.58   mIoU:0.5580
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp7 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp7.txt
# 最佳模型 mpa:88.55   mIoU:0.5585    当前跑的最佳模型
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_mixture_bits.py --mixture_16bit_8bit mixture_quant_config.txt -l 0.00001 -e 50 -b 8 --quarter_channel --strict_cin_number -q --OAQ_m 50 -f -M -r q -m ./checkpoint/ckpt_int16_int8_mixture_100epoch_20220222_lr0001_exp3/CP_best_epoch88.pth -c ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp8 2>&1 | tee ./checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp8.txt


################################################## eval quantization stage2 ####################################################
python eval.py -b 8 --mixture_16bit_8bit mixture_quant_config.txt --quarter_channel -t qfm --inference_type full_int -m checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp8/CP_best_epoch4.pth

################################################## save integer model ###############################################################
python eval.py -b 2 --mixture_16bit_8bit mixture_quant_config.txt --quarter_channel -t qfm --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int16_int8_mixture_50epoch_20220222_stage2_exp8/CP_best_epoch4.pth --save_model --result_name data/Unet_output_int16_int8_mixture_quarter_channel_strict

################################################## Load Integer model ###############################################################
python load_scale_and_param.py --scale_table_file data/Unet_output_int16_int8_mixture_quarter_channel_strict.scale --quantized_param_file data/Unet_output_int16_int8_mixture_quarter_channel_strict_param.npy
