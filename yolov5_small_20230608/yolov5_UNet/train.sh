########################################## 正常训练 ##############################################################
python train_yolov5.py --weights yolov5s.pt --img 416 --data data/PCB_data.yaml --batch-size 8 --epochs 200


########################################## 稀疏训练 ##############################################################
python train_sparity.py --st --sr 0.0002 --weights "上一步正常训练的结果".pt --data data/PCB_data.yaml --epochs 200 --img 416 --adam
# 训练中需要通过tensorboard监控训练过程, 特别是map变化, bn分布变化等, 在runs/train/exp*/目录下有events.out.tfevents.* 文件


########################################## 剪枝 ##############################################################
python prune.py --percent 0.5 --weights "上一步稀疏训练的结果".pt --data data/PCB_data.yaml --img 416 
# 会保存剪枝后模型pruned_model.pt


########################################## 微调 ##############################################################
python finetune_prune.py --weights "上一步剪枝的结果".pt --data data/PCB_data.yaml --img 416 --epochs 200


########################################## evaluate fp32 ##############################################################
python val.py --weights weight_finetune/fp32.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python quan_train.py --weights weight_finetune/fp32.pt --img 416 --data data/PCB_data.yaml --batch-size 8
########################################## evaluate full_int ##############################################################
python eval_quan.py --weights weight_finetune/w4a4b16.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python eval_quan.py --weights weight_finetune/w8a8b16.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python eval_quan.py --weights weight_finetune/w14a14b32.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python eval_quan.py --weights weight_finetune/w16a16b32.pt --img 416 --data data/PCB_data.yaml --batch-size 8

e
################################################### save_quantizd_layer ###############################################################
python eval_quan.py --weights weight_finetune/w4a4b16.pt --save_quantized_layer --result_name output_full_int/w4a4b16

python eval_quan.py --weights weight_finetune/w8a8b16.pt --save_quantized_layer --result_name output_full_int/w8a8b16

python eval_quan.py --weights weight_finetune/w14a14b32.pt --save_quantized_layer --result_name output_full_int/w14a14b32

python eval_quan.py --weights weight_finetune/w16a16b32.pt --save_quantized_layer --result_name output_full_int/w16a16b32


############################# load quantized layer #########################################
python load_quan_scale_param.py --scale_table_file output_full_int/w4a4b16.scale --quantized_param_file output_full_int/w4a4b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w8a8b16.scale --quantized_param_file output_full_int/w8a8b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w14a14b32.scale --quantized_param_file output_full_int/w14a14b32_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w16a16b32.scale --quantized_param_file output_full_int/w16a16b32_param.npy

