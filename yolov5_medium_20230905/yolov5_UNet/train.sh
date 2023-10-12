########################################## 正常训练 ##############################################################
python train_yolov5.py --weights yolov5m.pt --img 416 --data data/traffic_data.yaml --batch-size 8 --epochs 200

########################################## 稀疏训练 ##############################################################
# 训练中需要通过tensorboard监控训练过程, 特别是map变化, bn分布变化等, 在runs/train/exp*/目录下有events.out.tfevents.* 文件
python train_sparity.py --st --sr 0.0002 --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/exp5/weights/best.pt --data data/traffic_data.yaml --epochs 200 --img 416 --adam

########################################## 剪枝 ##############################################################
# 会保存剪枝后模型pruned_model.pt
python prune.py --percent 0.7 --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/exp12/weights/best.pt --data data/traffic_data.yaml --img 416

########################################## 微调 ##############################################################
CUDA_VISIBLE_DEVICES=1 python finetune_prune.py --weights pruned_model.pt --data data/traffic_data.yaml --img 416 --epochs 200



########################################## train quant ##############################################################

CUDA_VISIBLE_DEVICES=1,2 python quan_train.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/exp14/weights/best.pt --img 416 --data data/traffic_data.yaml --batch-size 8 --quantization_bits 4 --bias_bits 16 --name w4a4b16
CUDA_VISIBLE_DEVICES=1,2 python quan_train.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/exp14/weights/best.pt --img 416 --data data/traffic_data.yaml --batch-size 8 --quantization_bits 8 --bias_bits 16 --name w8a8b16
CUDA_VISIBLE_DEVICES=1,2 python quan_train.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/exp14/weights/best.pt --img 416 --data data/traffic_data.yaml --batch-size 8 --quantization_bits 16 --bias_bits 32 --name w16a16b32
########################################## evaluate full_int ##############################################################
CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w4a4b16/weights/best.pt --img 416 --data data/traffic_data.yaml  --batch-size 8 --name w4a4b16 --exist-ok

CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w8a8b16/weights/best.pt --img 416 --data data/traffic_data.yaml  --batch-size 8 --name w8a8b16 --exist-ok

CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w16a16b32/weights/best.pt --img 416 --data data/traffic_data.yaml  --batch-size 8 --name w16a16b32 --exist-ok

################################################### save_quantizd_layer ###############################################################

CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w4a4b16/weights/best.pt --save_quantized_layer --result_name output_full_int/w4a4b16


CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w8a8b16/weights/best.pt --save_quantized_layer --result_name output_full_int/w8a8b16

CUDA_VISIBLE_DEVICES=1,2 python eval_quan.py --weights /mnt/cephfs/home/lyy/Quantization/yolov5_medium_20230905/yolov5_UNet/runs/train/w16a16b32/weights/best.pt --save_quantized_layer --result_name output_full_int/w16a16b32



############################# load quantized layer #########################################
python load_quan_scale_param.py --scale_table_file output_full_int/w4a4b16.scale --quantized_param_file output_full_int/w4a4b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w8a8b16.scale --quantized_param_file output_full_int/w8a8b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w14a14b32.scale --quantized_param_file output_full_int/w14a14b32_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w16a16b32.scale --quantized_param_file output_full_int/w16a16b32_param.npy

