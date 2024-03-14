python predict.py  --bits 4 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_032811/best/best.pth'  --img_path '../pcb/3/65_i.jpg' >out_int4_pred.txt

python predict.py  --bits 8 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_032811/best/best.pth' --img_path '../pcb/3/58_i.jpg' >out_int8_pred.txt

python predict.py  --bits 16 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_032811/best/best.pth'  --img_path '../pcb/3/58_i.jpg' >out_int16_pred.txt


#python predict.py  --bits 4 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_020715/best/best.pth'  --img_path '../pcb/3/259_i.jpg' >out_int4_pred.txt

#python predict.py  --bits 8 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_020702/best/best.pth' --img_path '../pcb/3/259_i.jpg' >out_int8_pred.txt

#python predict.py  --bits 16 --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_020911/best/best.pth'  --img_path '../pcb/3/259_i.jpg' >out_int16_pred.txt

python predict.py  --bits 4 --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121223/best/best.pth'  --img_path '../pcb_3/3/58_i.jpg' >out_int4_pred.txt

python predict.py  --bits 8 --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121312/best/best.pth' --img_path '../pcb_3/3/58_i.jpg' >out_int8_pred.txt

python predict.py  --bits 16 --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121318/best/best.pth'  --img_path '../pcb_3/3/58_i.jpg' >out_int16_pred.txt


python predict.py  --bits 16 --mode eval --data /mnt/cephfs/home/lyy/data/body_detect/split_data  --config config/res18/retrain-pcb-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch   --model_path /mnt/cephfs/home/lyy/Quantization/dmcp_body/stage3/int16/AdaptiveResNet18_1040.0_030517/best/best.pth --img_path /mnt/cephfs/home/lyy/data/body_detect/split_data/train/0/0_1.jpg >out_int16_pred.txt

python predict.py  --bits 4 --mode eval --data /mnt/cephfs/home/lyy/data/body_detect/split_data  --config config/res18/retrain-pcb-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch   --model_path /mnt/cephfs/home/lyy/Quantization/dmcp_body/stage3/int4/AdaptiveResNet18_1040.0_030515/best/best.pth --img_path /mnt/cephfs/home/lyy/data/body_detect/split_data/train/0/0_1.jpg >out_int4_pred.txt
