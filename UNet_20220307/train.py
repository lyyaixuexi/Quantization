import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet import *
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import CityscapesDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tflite_quantization_PACT_weight_and_act as tf
from overflow_utils import *
import torchvision.transforms.functional as TF

# hook setting
def register_hook(model, func):
    # 注册钩子，用于获取每层卷积的输入和输出
    handle_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, tf.Conv2d_quantization):
            handle = layer.register_forward_hook(func)
            handle_list.append(handle)

    return handle_list

oaq_conv_result = []
def hook_conv_results(module, input, output):
    # 获取每层卷积的输出
    oaq_conv_result.append(output.detach().clone())


def hook_conv_results_checkoverflow(module, input, output):
    # conv_accumulator_bits: [min, max], sign=True
    # 检测是否累加器溢出
    bias_bits_global = tf.c_round(module.bias_bits)
    min = - 2 ** (bias_bits_global - 1) + 1
    max = 2 ** (bias_bits_global - 1) - 1
    if isinstance(module, tf.Conv2d_quantization):
        scale = (module.act_scale * module.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            output)
    output = tf.c_round(output.detach() / scale)
    down_overflow_index = output < min
    up_overflow_index = output > max
    No = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index))

    if No>0:
        print("############################ overflow happen ###################")
        print("overflow No: {}".format(No))
        print("overflow module: {}".format(module))
        print("module.alpha: {}".format(module.alpha))
        print("module.act_scale: {}".format(module.act_scale))
        print("output.min: {}".format(output.min()))
        print("output.max: {}".format(output.max()))
        print("max: {}".format(max))


def train_net(net,
              local_rank,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              args=None):

    train = CityscapesDataset('/home/lyy/data/Cityscapes',split='train', mode='fine',augment=True)
    val = CityscapesDataset('/home/lyy/data/Cityscapes', split='val', mode='fine', augment=False)
    n_train=train.__len__()
    train_sampler=DistributedSampler(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    if dist.get_rank() == 0:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Checkpoints:     {save_cp}
        ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)

    if net.module.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    if args.check_overflow:
        # Set hook to store Conv results for No
        print("Set hook to store Conv results for No")
        handle_list = register_hook(net.module, hook_conv_results_checkoverflow)


    logging.info('eval before training')
    if dist.get_rank() == 0:
        mIoU = eval_net(net=net.module, loader=val_loader, device=local_rank, input_scale=1)


    best_mIoU = 0

    global_step = 0
    for epoch in range(epochs):
        net.train()
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0

        if dist.get_rank() == 0:
            pbar=tqdm(total=len(train_loader)*batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img')

        for image, targetmask in train_loader:
            if (args.OAQ_m!=0) and (global_step%args.OAQ_m==0):
                oaq_handle_list = register_hook(net.module, hook_conv_results)

            imgs, true_masks = image, targetmask

            assert imgs.shape[1] == net.module.n_channels, \
                f'Network has been defined with {net.module.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(local_rank, dtype=torch.float32)
            mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
            true_masks = true_masks.to(local_rank, dtype=mask_type)

            if args.Mn_aware:
                tf.replace_next_act_scale(net.module)

            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            if dist.get_rank() == 0:
                writer.add_scalar('Loss_batch/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.module.parameters(), 0.1)
            # print(net.module.inc.double_conv[0].weight.grad)

            optimizer.step()

            if dist.get_rank() == 0:
                pbar.update(imgs.shape[0])

            if (args.OAQ_m!=0) and (global_step%args.OAQ_m==0):
                with torch.no_grad():
                    # if dist.get_rank() == 0:
                    #     logging.info("step={}, calculate No and updata alpha!".format(global_step))
                    No = calculate_No(local_rank, net.module, oaq_conv_result, logging)
                    total_No = tf.Gather.apply(No)
                    # print('total_No={}'.format(total_No))
                    No = torch.sum(total_No, dim=0)
                    # print('No={}'.format(No))
                    if torch.sum(No).item()!=0 and dist.get_rank()==0:
                        print('overflow still exists')
                    lr_max = lr
                    lr_curr = optimizer.param_groups[0]['lr']
                    update_alpha(local_rank, net.module, No, batch_size, lr_max, lr_curr, logging)

                for handle in oaq_handle_list:
                    handle.remove()

                oaq_conv_result.clear()

            global_step += 1

        if args.check_overflow:
            for handle in handle_list:
                handle.remove()

        if dist.get_rank() == 0:
            pbar.close()
            writer.add_scalar('Loss_epoch/train', epoch_loss, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(epoch_loss)

        # eval
        if dist.get_rank() == 0:
            mIoU = eval_net(net=net.module,loader=val_loader,device=local_rank,input_scale=1)
            if best_mIoU < mIoU:
                best_mIoU = mIoU

                if save_cp and dist.get_rank()==0:
                    # try:
                    #     os.mkdir(args.checkpoint)
                    #     logging.info('Created checkpoint directory')
                    # except OSError:
                    #     pass

                    if not os.path.exists(args.checkpoint):
                        os.mkdir(args.checkpoint)
                        logging.info('Created checkpoint directory')

                    torch.save(net.module.state_dict(),
                               args.checkpoint + f'/CP_best_epoch{epoch + 1}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')

            logging.info('best_mIoU: {}'.format(best_mIoU))

    if dist.get_rank() == 0:
        writer.close()

    return


def eval_net(net, loader, device, input_scale, inference_type=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    pix_tot=0
    I = [0] * net.n_classes
    U = [0] * net.n_classes
    dice_I = [0] * net.n_classes
    dice_U = [0] * net.n_classes

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for image, targetmask in loader:
            imgs, true_masks = image, targetmask
            imgs = imgs.to(device=device, dtype=torch.float32)
            if inference_type=='full_int':
                imgs=imgs/input_scale
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                # tot += F.cross_entropy(mask_pred, true_masks).item()

                # class_mask=mask_pred.argmax(dim=1)
                # class_mask = class_mask.unsqueeze(0).float()
                # class_mask = F.interpolate(class_mask, size=[1024, 2048], mode='nearest')
                # class_mask = class_mask.squeeze(0)

                class_mask = mask_pred.argmax(dim=1)
                class_mask = TF.resize(class_mask, size=(1024, 2048), interpolation=TF.InterpolationMode.NEAREST)


            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

            for i in range(net.n_classes):
                I[i]+=torch.sum(torch.where((class_mask == i) * (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()
                dice_I[i]+=2*torch.sum(torch.where((class_mask == i) * (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()

                U[i]+=torch.sum(torch.where((class_mask == i) + (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()
                dice_U[i]+=torch.sum(torch.where(class_mask == i,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()+torch.sum(torch.where(true_masks == i,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()

            pix_batch=torch.sum(torch.where(class_mask==true_masks,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()

            pix_tot+=pix_batch
            pbar.update()

    mpa = pix_tot / n_val / true_masks.size()[0] / true_masks.size()[1] / true_masks.size()[2]
    logging.info('mpa: {}%'.format(mpa*100))
    logging.info('print IoU:')
    K=0
    IoU_tot=0
    for i in range(net.n_classes):
        if U[i]!=0:
            K+=1
            IoU_tot += I[i]/U[i]
            print('{}:{}'.format(i,I[i]/U[i]))
        else:
            print('class {} not exist'.format(i))
    print('mIoU: {}\n'.format(IoU_tot/K))
    # logging.info('print dice:\n')
    # for i in range(net.n_classes):
    #     print('{}:{}'.format(i,dice_I[i]/dice_U[i] if dice_U[i]!=0 else 0))

    net.train()

    return IoU_tot/K



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')

    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument('-q', '--quantize', action='store_true',
                        help="quantization", default=False)
    parser.add_argument('-f', '--fuseBN', action='store_true',
                        help="fuseBN", default=False)
    parser.add_argument('-M', '--Mn_aware', action='store_true',
                        help="Mn_aware", default=False)
    parser.add_argument('-x', '--quantization_bits', metavar='QB', type=int, nargs='?', default=6,
                        help='quantization_bits', dest='quantization_bits')
    parser.add_argument('-y', '--m_bits', metavar='MB', type=int, nargs='?', default=12,
                        help='m_bits', dest='m_bits')
    parser.add_argument('-z', '--bias_bits', metavar='BB', type=int, nargs='?', default=16,
                        help='bias_bits', dest='bias_bits')

    parser.add_argument('--check_overflow', action='store_true',
                        help='if true, check the overflow of convolution before training')
    parser.add_argument('--OAQ_m', type=int, default=0)

    parser.add_argument('--model', '-m', default=None,
                        metavar='FILE')
    parser.add_argument('-r', '--resume', type=str, default='',
                        help='type of resume')

    parser.add_argument('--checkpoint', '-c', default='checkpoint_name',
                        metavar='DIRECTORY')

    parser.add_argument('--half_channel', action='store_true',
                        help='if true, use half_channel')

    parser.add_argument('--quarter_channel', action='store_true',
                        help='if true, use quarter_channel')

    parser.add_argument('--strict_cin_number', action='store_true',
                        help='if true, use half_channel')

    parser.add_argument('--reset_M', action='store_true',
                        help='if true, reset the M after load model')

    # todo
    parser.add_argument('--channel_prune_rate', type=float, nargs='?', default=0,
                        help='how many percentage of channel should be pruned, make sure that C_in*k*k<4096 and the channel numbers are divisible by 32')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=20, nearest=True, half_channel=args.half_channel, quarter_channel=args.quarter_channel, strict_cin_number=args.strict_cin_number)

    # keys = ["total", "==0", 10, 8, 6, 5, 4, 3, 2]
    # len_of_keys = len(keys)
    # param_numbers = torch.zeros([len_of_keys])
    # for name, param in net.named_parameters():
    #     if "pruned" in name:
    #         print("ignore {}".format(name))
    #         continue
    #     else:
    #         print("name:{}".format(name))
    #     param_num = param.numel()
    #     param_numbers[0] += param_num
    #
    #     param_equal_zero = param.eq(0).sum().item()
    #     param_numbers[1] += param_equal_zero
    #
    #     for key_index in range(2, len_of_keys):
    #         key = keys[key_index]
    #         threashold = torch.pow(torch.Tensor([0.1]), int(key))
    #         param_abs_bigger_or_equal_than_threashold = param.abs().ge(threashold).sum().item()
    #         param_abs_smaller_than_threashold = param_num - param_abs_bigger_or_equal_than_threashold
    #         param_numbers[key_index] += param_abs_smaller_than_threashold
    #         print("key:{} threashold:{} param_smaller_than_threashold:{}".format(key, threashold,
    #                                                                              param_abs_smaller_than_threashold))
    #
    # for i in range(len_of_keys):
    #     print("key:{} number:{}M ratio:{}%".format(keys[i], param_numbers[i] / 1000 / 1000,
    #                                                100 * param_numbers[i] / param_numbers[0]))


    if dist.get_rank() == 0:
        logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Nearest" if net.nearest else "Transposed conv"} upscaling')
        if args.model and args.resume=='':
            net.load_state_dict(torch.load(args.model))
            logging.info(f'Model loaded from {args.model}')
            print(net)

    if args.quantize:
        net = tf.replace(net, args.quantization_bits, args.m_bits, args.bias_bits, "all_fp", False, False)

        if args.model and args.resume == 'q' and dist.get_rank() == 0:
            net.load_state_dict(torch.load(args.model))
            logging.info(f'Model loaded from {args.model}')
            print(net)

    if args.fuseBN:
        net = tf.fuse_doubleconv(net)
        if args.model and args.resume == 'f' and dist.get_rank() == 0:
            net.load_state_dict(torch.load(args.model))
            logging.info(f'Model loaded from {args.model}')
            print(net)

    if args.Mn_aware:
        net = tf.open_Mn(net, args.m_bits)
        if args.model and args.resume == 'm' and dist.get_rank() == 0:
            net.load_state_dict(torch.load(args.model))
            logging.info(f'Model loaded from {args.model}')

            print(net)

        if args.reset_M:
            print("reset_M is True!")
            for name, module in net.named_modules():
                if isinstance(module, tf.Conv2d_quantization):
                    module.m_bits = module.m_bits - module.m_bits + args.m_bits
                    module.max_M = module.max_M - module.max_M + tf.c_round(2 ** module.m_bits - 1)

    if dist.get_rank() == 0:
        print(net)

    net=net.to(local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net=DDP(net, device_ids=[local_rank], output_device=local_rank)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  local_rank=local_rank,
                  args=args)
    except KeyboardInterrupt:
        if dist.get_rank() == 0:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
