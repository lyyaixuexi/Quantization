export PYTHONPATH=$PYTHONPATH:/userhome/resnet18_fault_det/dmcp-master
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py --OAQ_m 10 --mode train --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2-v2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 6 --m_bits 12 --bias_bits 16 --fuseBN 'stage1/int8/AdaptiveResNet18_1040.0_121308/best/best.pth'  --save_dir './stage2/int8' >out_8bits_stage2.txt

#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py  --mode train --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 6 --m_bits 12 --bias_bits 16 --fuseBN 'stage1/int8/AdaptiveResNet18_1040.0_120818/best/best.pth'  --save_dir './stage2/int8' >out_8bits_stage2.txt

CUDA_VISIBLE_DEVICES=3 python train_quantize.py --mode train  --data ../pcb_3 --config config/res18/retrain-tsr-stage2-v2.yaml --flops 1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch --quantization --quantization_bits 8 --m_bits 12 --bias_bits 16 --fuseBN /mnt/cephfs/home/lyy/Quantization/dmcp/stage1/int8/AdaptiveResNet18_1040.0_022116/best/best.pth --save_dir './stage2/int8' >out_8bits_stage2.txt

CUDA_VISIBLE_DEVICES=0 python train_quantize.py --mode evaluate  --data ../pcb_3 --config config/res18/retrain-tsr-stage2-v2.yaml --flops 1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch --quantization --quantization_bits 6 --m_bits 12 --bias_bits 16 --fuseBN /mnt/cephfs/home/lyy/Quantization/dmcp/stage1/int8/AdaptiveResNet18_1040.0_010219/best/best.pth --save_dir './stage2/int8' >out_8bits_stage2_eval.txt

