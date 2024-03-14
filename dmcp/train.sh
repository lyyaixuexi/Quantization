#pruning
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --data ../pcb  --config config/res18/dmcp-pcb.yaml --flops  2080

#retrain/finetuning:./results/DMCPResNet18_1040.0_042616/checkpoints/0426_1942.pth
#python main.py --mode train --data ../pcb  --config 'config/res18/retrain-pcb.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'


CUDA_VISIBLE_DEVICES=1 python main.py --mode train --data ../pcb  --config config/res18/dmcp-tsr.yaml --flops  2080

CUDA_VISIBLE_DEVICES=3 python main.py --mode train --data ../pcb  --config config/res18/retrain-tsr.yaml --flops 250 --chcfg '/mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_250.0_122214/model_sample/expected_ch'

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --data ../pcb  --config config/res18/retrain-tsr-v3.yaml --flops  1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch


CUDA_VISIBLE_DEVICES=4 python train_quantize.py --mode train --data ../pcb_3 --config config/res18/retrain-tsr-v3.yaml --flops 1040 --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch --quantization  --quantization_bits 32 --m_bits 16 --bias_bits 32 --save_dir './stage1/int32' >out_32bits_stage1.txt
