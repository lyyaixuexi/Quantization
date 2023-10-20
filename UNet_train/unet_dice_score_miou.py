import os
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from utils.data_loading import CarvanaDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet import UNet
from utils.metrics import Evaluator
import cv2


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    n = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch'):
        image, mask_true, file_name = batch['image'], batch['mask'], batch['name']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            mask_pred = net(image)
            cv2.imwrite(f"output/{image.shape[-1]}/{file_name[0]}.png", mask_pred.argmax(1).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 50)
            n += 1

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # print(mask_pred)
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

def compute_mIoU(net, dataloader, device):
    # eval
    net.eval()
    num_classes = net.n_classes
    evaluator = Evaluator(num_classes)
    evaluator.reset()
    n = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, desc='mIoU1', unit='batch'):
        #   读取每一个（图片-标签）对
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)   # input:float32, weight:float32
        mask_true = mask_true.to(device=device, dtype=torch.long)  # 标签：1*256*256  int64

        with torch.no_grad():
            # 模型前推得到图像分割结果，转化成numpy数组
            mask_pred = net(image)  # 1*15*256*256 float32

            n += 1
            mask_pred = mask_pred.cpu().numpy()
            mask_true = mask_true.cpu().numpy()
            mask_pred = np.argmax(mask_pred, axis=1)
            evaluator.add_batch(mask_true, mask_pred)

    #   创建一个混淆矩阵
    confusion_matrix = evaluator.confusion_matrix
    print(confusion_matrix)
    # acc = evaluator.Pixel_Accuracy()
    mPA = evaluator.Pixel_Accuracy_Class()
    mean_iou = evaluator.Mean_Intersection_over_Union()

    logging.info('confusion_matrix:')
    logging.info(f"val mIoU: {mean_iou}")
    logging.info(f"val mPA: {mPA}")


    # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> mIoU: ' + str(mean_iou) + '; mPA: ' + str(mPA))
    return mean_iou


if __name__ == "__main__":

    img_scale = 0.5
    batch_size = 1
    bilinear = True
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Create dataset ISAID
    dir_img_eval = Path('/home/liaoyuyan/dataset/ISAID/val/val_images')
    dir_mask_eval = Path('/home/liaoyuyan/dataset/ISAID/val/Semantic_masks/images')
    dataset_eval = CarvanaDataset(dir_img_eval, dir_mask_eval, img_scale)
    n_val = len(dataset_eval)
    val_set = Subset(dataset_eval, range(n_val))
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=4, pin_memory=True)

    logging.info(f'''Starting evaluate:
            Batch size:      {batch_size}
            Validation size: {n_val}
            Images scaling:  {img_scale}
        ''')

    # 2. Load pretrained model.
    bilinear = True
    net = UNet(n_channels=3, n_classes=15, bilinear=bilinear)
    netPath = '/home/liaoyuyan/Quantization/UNet_train/checkpoints/ISAID/checkpoint512_withCE_epoch50.pth'
    print('==> load pretrained UNet model..')
    assert os.path.isfile(netPath), 'Error: no checkpoint directory found!'
    ch = torch.load(netPath)
    net.load_state_dict(ch, strict=False)
    net.to(device=device)
    net.eval()

    # 3. compute Dice_score
    Dice_score = evaluate(net, val_loader, device)
    logging.info('Validation Dice_score: {}'.format(Dice_score))

    # 4. compute mIoU
    mIoU = compute_mIoU(net, val_loader, device)
    from evaluate import compute_mIoU as miou
    mIoU2 = miou(net, val_loader, device)
    logging.info('Validation mIoU: {}'.format(mIoU))
    logging.info('Validation mIoU2: {}'.format(mIoU2))
