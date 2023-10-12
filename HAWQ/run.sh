CUDA_VISIBLE_DEVICES=1 python quant_train.py -a resnet18 \
                                             --epochs 1 \
                                             --lr 0.0001 \
                                             --batch-size 128 \
                                             --data /mnt/ssd/datasets/imagenet/ \
                                             --pretrained \
                                             --save-path resnet18_test \
                                             --act-range-momentum=0.99 \
                                             --wd 1e-4 \
                                             --data-percentage 0.0001 \
                                             --fix-BN \
                                             --checkpoint-iter -1 \
                                             --quant-scheme uniform8

CUDA_VISIBLE_DEVICES=1 python quant_train.py -a resnet18 \
                                             --epochs 1 \
                                             --lr 0.0001 \
                                             --batch-size 128 \
                                             --data /mnt/ssd/datasets/imagenet/ \
                                             --pretrained \
                                             --save-path resnet18_test \
                                             --act-range-momentum=0.99 \
                                             --wd 1e-4 \
                                             --data-percentage 0.0001 \
                                             --fix-BN \
                                             --checkpoint-iter -1 \
                                             --quant-scheme uniform4

CUDA_VISIBLE_DEVICES=1 python quant_train.py -a resnet18 \
                                             --epochs 1 \
                                             --lr 0.0001 \
                                             --batch-size 128 \
                                             --data /mnt/ssd/datasets/imagenet/ \
                                             --pretrained \
                                             --save-path resnet18_test \
                                             --act-range-momentum=0.99 \
                                             --wd 1e-4 \
                                             --data-percentage 0.0001 \
                                             --fix-BN \
                                             --checkpoint-iter -1 \
                                             --quant-scheme bops_0.5

