python model_transform.py --bits 4   --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121223/best/best.pth'  --result_name 'data/output_int4' >model_transform_int4.txt

python model_transform.py --bits 8  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121312/best/best.pth'  --result_name 'data/output_int8'  >model_transform_int8.txt

python model_transform.py --bits 16   --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121318/best/best.pth'  --result_name 'data/output_int16'  >model_transform_int16.txt

#python model_transform.py --bits 4   --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_032811/best/best.pth'  --result_name 'data/output_int4' >model_transform_int4.txt

#python model_transform.py --bits 8  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_032811/best/best.pth'  --result_name 'data/output_int8'  >model_transform_int8.txt

#python model_transform.py --bits 16   --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_032811/best/best.pth'  --result_name 'data/output_int16'  >model_transform_int16.txt


python model_transform.py --bits 4   --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch  --model_path /home/lyy/Quantization/dmcp_2024_02_29/stage3/4/best.pth  --result_name data/output_int4 >model_transform_int4.txt

python model_transform.py --bits 8  --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch  --model_path /home/lyy/Quantization/dmcp_2024_02_29/stage3/8/best.pth  --result_name data/output_int8  >model_transform_int8.txt

python model_transform.py --bits 16   --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch  --model_path /home/lyy/Quantization/dmcp_2024_02_29/stage3/16/best.pth  --result_name data/output_int16  >model_transform_int16.txt
