# 云脑容器
dockerhub.pcl.ac.cn:5000/user-images/lihk:pt18_deforable_detr_timm

# 云脑运行指令
bash
cd /userhome/UNet && xxxx

# 指标：
# MIOU(%)：mean intersection over union （均交并比）
# IOU = 正例预测成正例 /（正例预测成正例+正例预测成反例+反例预测成正例）=（target⋀prediction）/（target⋃prediction）

# 加载原模型，在测试集上计算性能指标。
# 模型大小: 65.955 MB
# mIoU(%): 59.12
python eval.py -b 4 -m checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth

# 加载压缩后的模型，在测试集上计算性能指标。
# 模型大小:  MB
# mIoU(%): 58.30
python eval.py -b 2 --half_channel -t qfm -x 8 -y 12 -z 16 --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211010_stage2/CP_best_epoch4.pth

# save model parameter from fp32 to int64
python eval.py -b 2 -t qfm -x 8 -y 12 -z 16 --half_channel --strict_cin_number --inference_type full_int -m checkpoint/ckpt_int8_50epoch_20211019_stage2/CP_best_epoch8.pth --save_model --result_name data/Unet_output_int8_half_channel_strict
# compress model parameter from int64 to low-bitwidth int
python convert_parameters_from_fp32_to_int.py --fp32_quantized_param_file data/Unet_output_int8_half_channel_strict_param.npy