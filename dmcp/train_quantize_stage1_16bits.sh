export PYTHONPATH=$PYTHONPATH:/userhome/resnet18_fault_det/dmcp-master
 
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py  --mode train --data ../pcb_3  --config 'config/res18/retrain-pcb-v2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 14 --m_bits 12 --bias_bits 32 --save_dir './stage1/int16' >out_16bits_stage1.txt

CUDA_VISIBLE_DEVICES=4 python train_quantize.py --mode train --data ../pcb_3 --config config/res18/retrain-tsr-v3.yaml --flops 1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch --quantization  --quantization_bits 16 --m_bits 12 --bias_bits 32  --save_dir './stage1/int16' >out_16bits_stage1.txt

CUDA_VISIBLE_DEVICES=4 python train_quantize.py --mode evaluate --data ../pcb_3 --config config/res18/retrain-tsr-v3.yaml --flop s 1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch  --quantization  --quantization_bits 14 --m_bits 12 --bias_bits 32 --save_dir './stage1/int16' >out_16bits_stage1_eval.txt
