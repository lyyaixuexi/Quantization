#not prune:raw
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --raw --mode eval --data ../pcb  --config 'config/res18/dmcp-pcb.yaml' --flops  1040  --model_path 'results/AdaptiveResNet18_1040.0_051315/best/best.pth'  >out_raw_test.txt
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --raw --mode eval --data ../pcb  --config 'config/res18/dmcp-pcb.yaml' --flops  1040  --model_path 'results/DMCPResNet18_1040.0_051312/best/best.pth'  >out_raw_test.txt

#适配硬件的量化截断方案(仅在输出截断一次，输入不截断)，4/8/16.根据领域惯例，网络在4bit量化时第一层用的8bit,并将输出截断到4bit(stage3)
#int4
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121223/best/best.pth'  >out_int4_test.txt

#int8
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int8_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121312/best/best.pth'  >out_int8_test.txt

#int16
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int16_test.txt
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121223/best/best.pth'  >out_int16_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121318/best/best.pth'  >out_int16_test.txt

#############################以下是在PCB 3个类上训练的结果目录,epochs=300,4/6/15 #############################
#int4
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121018/best/best.pth'  >out_int4_test.txt

#int8
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int8_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121018/best/best.pth'  >out_int8_test.txt

#int16
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int16_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121016/best/best.pth'  >out_int16_test.txt


#############################以下是在PCB 3个类上训练的结果目录,epochs=100,4/8/16, 8bit还存在一个溢出, #############################
#int4
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121016/best/best.pth'  >out_int4_test.txt

#int8
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int8_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121016/best/best.pth'  >out_int8_test.txt

#int16
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int16_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_3  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121016/best/best.pth'  >out_int16_test.txt




###################以下是在PCB 7个类上训练的结果目录####################################
#int4 
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python  test.py  --mode eval --data ../pcb_7  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int4/AdaptiveResNet18_1040.0_121013/best/best.pth'  >out_int4_test.txt

#int8:6bit
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int8_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_7  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int8/AdaptiveResNet18_1040.0_121013/best/best.pth'  >out_int8_test.txt

#int16:12bit
#/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_120723/best/best.pth'  >out_int16_test.txt
/userhome/anaconda3_1/envs/pytorch_wzb/bin/python   test.py  --mode eval --data ../pcb_7  --config 'config/res18/retrain-pcb-stage2.yaml' --flops  1040  --chcfg './results/DMCPResNet18_1040.0_051312/model_sample/expected_ch'  --model_path 'stage3/int16/AdaptiveResNet18_1040.0_121013/best/best.pth'  >out_int16_test.txt


CUDA_VISIBLE_DEVICES=4 python  test.py  --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-stage2.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch   --model_path /mnt/cephfs/home/lyy/Quantization/dmcp/stage3/int4/AdaptiveResNet18_1040.0_033017/best/best.pth >out_int4_test.txt

CUDA_VISIBLE_DEVICES=4 python  test.py  --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-v3.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch   --model_path /mnt/cephfs/home/lyy/Quantization/dmcp/stage3/int8/AdaptiveResNet18_1040.0_022801/best/best.pth >out_int8_test.txt

CUDA_VISIBLE_DEVICES=4 python  test.py  --mode eval --data ../pcb_3  --config config/res18/retrain-tsr-v3.yaml --flops  1040  --chcfg /mnt/cephfs/home/lyy/Quantization/dmcp/results/DMCPResNet18_1040.0_051312/model_sample/expected_ch   --model_path /mnt/cephfs/home/lyy/Quantization/dmcp/stage3/int16/AdaptiveResNet18_1040.0_022801/best/best.pth >out_int16_test.txt


