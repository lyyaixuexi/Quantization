export PYTHONPATH=$PYTHONPATH:/userhome/resnet18_fault_det/dmcp-master

#以下是在PCB 3个类上训练的命令
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py  --mode train --data ../pcb_3  --config 'config/res18/retrain-pcb-stage3-v2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 4 --m_bits 12 --bias_bits 16 --fuseBN 'stage1/int4/AdaptiveResNet18_1040.0_121206/best/best.pth'  --Mn_aware --load_pretrain_fuseBN 'stage2/int4/AdaptiveResNet18_1040.0_121220/best/best.pth' --save_dir './stage3/int4' >out_4bits_stage3.txt
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py  --mode train --data ../pcb_3  --config 'config/res18/retrain-pcb-stage3-v2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 4 --m_bits 12 --bias_bits 16 --fuseBN 'stage1/int4/AdaptiveResNet18_1040.0_120714/best/best.pth'  --Mn_aware --load_pretrain_fuseBN 'stage2/int4/AdaptiveResNet18_1040.0_120717/best/best.pth' --save_dir './stage3/int4' >out_4bits_stage3.txt


#以下是在PCB 7个类上训练的命令
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  train_quantize.py  --mode train --data ../pcb  --config 'config/res18/retrain-pcb-stage3.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch' --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  --quantization  --quantization_bits 4 --m_bits 12 --bias_bits 16 --fuseBN 'stage1/int4/AdaptiveResNet18_1040.0_120820/best/best.pth'  --Mn_aware --load_pretrain_fuseBN 'stage2/int4/AdaptiveResNet18_1040.0_120822/best/best.pth' --save_dir './stage3/int4' >out_4bits_stage3.txt



CUDA_VISIBLE_DEVICES=2 python train_quantize.py  --mode train --data ../pcb_3  --config config/res18/retrain-tsr-stage3-v2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch --quantization  --quantization_bits 4 --m_bits 12 --bias_bits 16 --fuseBN /mnt/cephfs/home/lyy/Quantization/dmcp/stage1/int4/AdaptiveResNet18_1040.0_022215/best/best.pth  --Mn_aware  --load_pretrain_fuseBN /mnt/cephfs/home/lyy/Quantization/dmcp/stage2/int4/AdaptiveResNet18_1040.0_022602/best/best.pth  --save_dir './stage3/int4' >out_4bits_stage3.txt

