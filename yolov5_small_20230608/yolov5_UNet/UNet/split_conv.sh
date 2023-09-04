# 云脑容器
dockerhub.pcl.ac.cn:5000/user-images/lihk:pt18_deforable_detr_timm

# 云脑运行指令
bash
cd /userhome/UNet && xxxx

# 指标：
# MIOU(%)：mean intersection over union （均交并比）
# IOU = 正例预测成正例 /（正例预测成正例+正例预测成反例+反例预测成正例）=（target⋀prediction）/（target⋃prediction）

# 加载原模型，在测试集上计算性能指标。 绘制原模型卷积层输出的特征图的分布直方图
python eval_split.py -b 4 -m checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth --channel_each_group -1 --hist_save_dir hist_original

# 加载8比特溢出感知训练的量化模型，在测试集上计算性能指标。绘制融合BN层后，卷积层输出的特征图的分布直方图
python eval_split.py -b 4 -t qfm -x 8 -y 12 -z 16 --inference_type all_fp -m checkpoint/ckpt_int8_50epoch_20210922_stage2/CP_best_epoch2.pth --channel_each_group -1 --hist_save_dir hist_baseline_fuseBN

# 加载原模型，在测试集上计算性能指标。 绘制BN层输出的特征图的分布直方图
python eval_split.py -b 4 -m checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth --channel_each_group -1 --hist_save_dir hist_original_afterBN --BN_hook

# 加载原模型，在测试集上计算性能指标。 绘制原模型卷积层经过split后，卷积层输出的特征图的分布直方图
python eval_split.py -b 4 -m checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth --channel_each_group 4 --hist_save_dir hist_split_4


# 训练全精度的剪枝后的Unet
# fp32  200 epoch  lr0.0001  survive only quarter channel   mpa: 85.90%  mIoU: 53.68
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_split.py -l 0.0001 -e 200 -b 16 --quarter_channel --strict_cin_number -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20221123 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_20221123.txt

# 训练全精度的剪枝后的Unet，使用Split Conv
# fp32  200 epoch  lr0.0001  survive only quarter channel   channel_each_group 16  mpa: xxx%  mIoU: xxx   split_conv_bn_relu_sum    训练崩掉了
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_split.py -l 0.0001 -e 200 -b 16 --quarter_channel --strict_cin_number --channel_each_group 16 --conv_type split_conv_bn_relu_sum -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_20221123 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_20221123.txt

# fp32  200 epoch  lr0.0001  survive only quarter channel   channel_each_group 8  mpa: xxx%  mIoU: xxx    CUDA Out of Memory
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_split.py -l 0.0001 -e 200 -b 16 --quarter_channel --strict_cin_number --channel_each_group 8 -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group8_20221124 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group8_20221124.txt

# fp32  200 epoch  lr0.0001  survive only quarter channel   channel_each_group 16  mpa: xxx%  mIoU: xxx    split_conv_sum_bn_relu   todo running
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_split.py -l 0.0001 -e 200 -b 16 --quarter_channel --strict_cin_number --channel_each_group 16 --conv_type split_conv_sum_bn_relu -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_split_conv_sum_bn_relu_20221124 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_split_conv_sum_bn_relu_20221124.txt

# fp32  200 epoch  lr0.0001  survive only quarter channel   channel_each_group 16  mpa: xxx%  mIoU: xxx    split_conv_bn_sum_relu   todo running
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_split.py -l 0.0001 -e 200 -b 16 --quarter_channel --strict_cin_number --channel_each_group 16 --conv_type split_conv_bn_sum_relu -c ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_split_conv_bn_sum_relu_20221124 2>&1 | tee ./checkpoint/ckpt_fp32_200epoch_quarter_channel_strict_cin_number_channel_each_group16_split_conv_bn_sum_relu_20221124.txt