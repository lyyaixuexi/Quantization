python main.py --dataset cifar10 --arch vgg --depth 19 --filename vgg
python main.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --filename vgg
python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model vgg --filename vgg_after_prune
python main.py -amp_loss --refine vgg_after_prune --dataset cifar10 --arch vgg --depth 19 --epochs 160 --filename pruned_vgg

python main_modify.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch tsr --depth 19 --filename tsr
python tsrprune.py --dataset tsr --depth 19 --percent 0.7 --model tsr_modify --filename tsr_after_prune
python main_modify.py -amp_loss --refine tsr_after_prune_0.1 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.1

CUDA_VISIBLE_DEVICES=1 python main_modify.py -amp_loss --refine tsr_after_prune_0.15 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.15
CUDA_VISIBLE_DEVICES=2 python main_modify.py -amp_loss --refine tsr_after_prune_0.2 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.2
CUDA_VISIBLE_DEVICES=3 python main_modify.py -amp_loss --refine tsr_after_prune_0.3 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.3




CUDA_VISIBLE_DEVICES=3 python main_modify.py -sr -amp_loss --s 0.0001 --arch tsr --depth 9 --filename tsr_modify_part2
CUDA_VISIBLE_DEVICES=4 python main_modify.py -sr -amp_loss --s 0.0001 --arch tsr --depth 9 --filename tsr_modify_part2_resume_gpu010 --resume /mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/tsr_modify_gpu010.pth

python main_modify.py -amp_loss --refine tsr_after_prune_0.1 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.1_gpu010 --batch-size 512
CUDA_VISIBLE_DEVICES=1 python main_modify.py -amp_loss --refine tsr_after_prune_0.15 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.15_gpu010 --batch-size 512
CUDA_VISIBLE_DEVICES=2 python main_modify.py -amp_loss --refine tsr_after_prune_0.2 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.2_gpu010 --batch-size 512
CUDA_VISIBLE_DEVICES=3 python main_modify.py -amp_loss --refine tsr_after_prune_0.3 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.3_gpu010 --batch-size 512
CUDA_VISIBLE_DEVICES=5 python main_modify.py -sr -amp_loss --s 0.0001 --arch tsr --depth 9 --filename tsr_modify_gpu010 --batch-size 768 --epochs 30


python main_modify.py -amp_loss --refine tsr_after_prune_0.1_part2 --dataset tsr --arch tsr --epochs 160 --filename pruned_tsr_0.1_gpu010_part2 --batch-size 512



python main_modify.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch tsr --depth 19 --filename tsr --resume /mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/tsr.pth
python main_modify.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch tsr --depth 19 --filename tsr_test --resume /mnt/cephfs/home/lyy/Quantization/Network-Slimming/logs/tsr.pth


###################################quant######################################

CUDA_VISIBLE_DEVICES=4 python train_quan.py -amp_loss --refine tsr_after_prune_0.1 --quant pruned_tsr_0.1 --dataset tsr --arch tsr --epochs 160 --filename quant_tsr_0.1

CUDA_VISIBLE_DEVICES=0 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_gpu010 --quant pruned_tsr_0.1_gpu010 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_4bit_withoutMN --quantization_bits 4 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_withoutMN.txt
CUDA_VISIBLE_DEVICES=1 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_gpu010 --quant pruned_tsr_0.1_gpu010 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_8bit_withoutMN --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_withoutMN.txt
CUDA_VISIBLE_DEVICES=2 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_gpu010 --quant pruned_tsr_0.1_gpu010 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_16bit_withoutMN --quantization_bits 16 --bias_bits 32 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_withoutMN.txt

CUDA_VISIBLE_DEVICES=5 python train_quan.py -amp_loss --refine tsr_after_prune_0.1 --quant pruned_tsr_0.1 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_4bit_withoutMN_per100 --quantization_bits 4 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_withoutMN_per100.txt
CUDA_VISIBLE_DEVICES=6 python train_quan.py -amp_loss --refine tsr_after_prune_0.1 --quant pruned_tsr_0.1 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_8bit_withoutMN_per100 --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_withoutMN_per100.txt
CUDA_VISIBLE_DEVICES=7 python train_quan.py -amp_loss --refine tsr_after_prune_0.1 --quant pruned_tsr_0.1 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_16bit_withoutMN_per100 --quantization_bits 16 --bias_bits 32 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_withoutMN_per100.txt


CUDA_VISIBLE_DEVICES=0 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_4bit_withoutMNstate_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_4bit_MN --quantization_bits 4 --bias_bits 16 --lr 1e-4 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_MN.txt
CUDA_VISIBLE_DEVICES=1 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_8bit_withoutMNstate_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_8bit_MN --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_MN.txt
CUDA_VISIBLE_DEVICES=2 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_16bit_withoutMNstate_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_16bit_MN --quantization_bits 16 --bias_bits 32 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_MN.txt

CUDA_VISIBLE_DEVICES=5 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_4bit_withoutMN_per100state_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_4bit_MN_per100 --quantization_bits 4 --bias_bits 16 --lr 1e-4 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_MN_per100.txt
CUDA_VISIBLE_DEVICES=6 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_8bit_withoutMN_per100state_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_8bit_MN_per100 --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_MN_per100.txt
CUDA_VISIBLE_DEVICES=7 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_16bit_withoutMN_per100state_dict_model --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_16bit_MN_per100 --quantization_bits 16 --bias_bits 32 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_MN_per100.txt

CUDA_VISIBLE_DEVICES=6 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant pruned_tsr_0.1 --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_8bit_MN_per100 --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_MN_per100.txt

################################################################with part2##############################################
CUDA_VISIBLE_DEVICES=1 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_part2 --quant pruned_tsr_0.1_gpu010_part2_best --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_4bit_withoutMN_part2 --quantization_bits 4 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_withoutMN_part2.txt
CUDA_VISIBLE_DEVICES=2 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_part2 --quant pruned_tsr_0.1_gpu010_part2_best --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_8bit_withoutMN_part2 --quantization_bits 8 --bias_bits 16 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_withoutMN_part2.txt
CUDA_VISIBLE_DEVICES=3 python train_quan.py -amp_loss --refine tsr_after_prune_0.1_part2 --quant pruned_tsr_0.1_gpu010_part2_best --dataset tsr --arch tsr --epochs 25 --filename quant_tsr_0.1_16bit_withoutMN_part2 --quantization_bits 16 --bias_bits 32 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_withoutMN_part2.txt

CUDA_VISIBLE_DEVICES=1 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_4bit_withoutMN_part2_best --dataset tsr --arch tsr --epochs 20 --filename quant_tsr_0.1_4bit_MN_part2 --quantization_bits 4 --bias_bits 16 --lr 1e-4 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_4bit_MN_part2.txt
CUDA_VISIBLE_DEVICES=2 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_8bit_withoutMN_part2_best --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_8bit_MN_part2 --quantization_bits 8 --bias_bits 16 --lr 1e-4 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_8bit_MN_part2.txt
CUDA_VISIBLE_DEVICES=3 python train_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_16bit_withoutMN_part2_best --dataset tsr --arch tsr --epochs 5 --filename quant_tsr_0.1_16bit_MN_part2 --quantization_bits 16 --bias_bits 32 --lr 1e-4 2>&1 | tee ./checkpoint/ckpt_quant_tsr_0.1_16bit_MN_part2.txt







###############################################################full_int测试与保存参数############################################################
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_4bit_MNstate_dict_model --quantization_bits 4 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_4bit_MNstate_dict_model
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_8bit_MNstate_dict_model --quantization_bits 8 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_8bit_MNstate_dict_model
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_16bit_MNstate_dict_model --quantization_bits 16 --bias_bits 32 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_16bit_MNstate_dict_model

python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_4bit_MN_per100state_dict_model --quantization_bits 4 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_4bit_MN_per100state_dict_model
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_8bit_MN_per100state_dict_model --quantization_bits 8 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_8bit_MN_per100state_dict_model
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1 --quant quant_tsr_0.1_16bit_MN_per100state_dict_model --quantization_bits 16 --bias_bits 32 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_16bit_MN_per100state_dict_model


python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_4bit_MN_part2_best --quantization_bits 4 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_4bit_MN_part2
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_8bit_MN_part2_best --quantization_bits 8 --bias_bits 16 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_8bit_MN_part2
python test_quan.py -amp_loss -M --refine tsr_after_prune_0.1_part2 --quant quant_tsr_0.1_16bit_MN_part2_best  --quantization_bits 16 --bias_bits 32 --save_quantized_layer --filename output_full_int/quant_tsr_0.1_16bit_MN_part2


python inference.py -amp_loss -M --refine tsr_after_prune_0.1_gpu010 --quant quant_tsr_0.1_16bit_MNstate_dict_model --dataset tsr--archtsr --epochs 5 --filename quant_tsr_0.1_16bit_MN_test --quantization_bits 16 --bias_bits 32