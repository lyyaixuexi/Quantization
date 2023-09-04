########################################## evaluate fp32 ##############################################################
python val.py --weights weight_finetune/fp32.pt --img 416 --data data/PCB_data.yaml --batch-size 8


########################################## evaluate full_int ##############################################################
python eval_quan.py --weights weight_finetune/w4a4b16.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python eval_quan.py --weights weight_finetune/w8a8b16.pt --img 416 --data data/PCB_data.yaml --batch-size 8

python eval_quan.py --weights weight_finetune/w16a16b32.pt --img 416 --data data/PCB_data.yaml --batch-size 8


################################################## save_quantized_layer ###############################################################
python eval_quan.py --weights weight_finetune/w4a4b16.pt --save_quantized_layer --result_name output_full_int/w4a4b16

python eval_quan.py --weights weight_finetune/w8a8b16.pt --save_quantized_layer --result_name output_full_int/w8a8b16

python eval_quan.py --weights weight_finetune/w16a16b32.pt --save_quantized_layer --result_name output_full_int/w16a16b32


############################# load quantized layer #########################################
python load_quan_scale_param.py --scale_table_file output_full_int/w4a4b16.scale --quantized_param_file output_full_int/w4a4b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w8a8b16.scale --quantized_param_file output_full_int/w8a8b16_param.npy

python load_quan_scale_param.py --scale_table_file output_full_int/w16a16b32.scale --quantized_param_file output_full_int/w16a16b32_param.npy

