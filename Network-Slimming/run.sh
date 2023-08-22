python main.py --dataset cifar10 --arch vgg --depth 19 --filename vgg
python main.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --filename vgg
python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model vgg --filename vgg_after_prune
python main.py -amp_loss --refine vgg_after_prune --dataset cifar10 --arch vgg --depth 19 --epochs 160 --filename pruned_vgg

python main_modify.py -sr -amp_loss --s 0.00002 --dataset cifar10 --arch tsr --depth 19 --filename tsr
python tsrprune.py --dataset tsr --depth 19 --percent 0.7 --model tsr --filename tsr_after_prune