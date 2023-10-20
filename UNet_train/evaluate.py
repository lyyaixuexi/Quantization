import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import numpy as np
from torchvision.utils import save_image
import onnx
import onnxruntime



def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    n=0

    #导入onnx模型
    # onnxpath = './onnx_models/trainvalsplit/Unet_scale0.5_fixedinputsize_960.onnx'
    # ort_session = onnxruntime.InferenceSession(onnxpath)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch'):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        #保存image
        # save_image(image.detach().cpu().squeeze(0),"./output/{}.jpg".format(dataloader.dataset.dataset.ids[n]))

        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        #保存mask
        # save_image(mask_true.detach().cpu().squeeze(0), "./output/{}"+"_mask.gif".format(dataloader.dataset.dataset.ids[n]))

        with torch.no_grad():
            mask_pred = net(image)
            ##### torch.save(mask_pred,"./Output_txt/torchsave1.pth")
            n += 1

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)


    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    assert (a.shape == b.shape)  # a的形状和b的形状必须相等
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)  # float32->int64


def per_class_iu(hist):
    # return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)  # 0.123
    return np.diag(hist) / (np.sum(hist, axis=1) + np.sum(hist, axis=0) - np.diag(hist))  # 0.924


def per_class_PA(hist):
    # return np.diag(hist) / np.maximum(hist.sum(1), 1)
    return np.diag(hist) / hist.sum(axis=1)

def compute_mIoU(net, dataloader, device):
    net.eval()
    num_classes = net.n_classes
    n = 0
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签
    #   获得验证集图像分割结果
    # ------------------------------------------------#

    # iterate over the validation set
    for batch in tqdm(dataloader, desc='mIoU', unit='batch'):
        #   读取每一个（图片-标签）对
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)   # input:float32, weight:float32
        mask_true = mask_true.to(device=device, dtype=torch.long)  # 标签：1*256*256  int64

        with torch.no_grad():
            # 模型前推得到图像分割结果，转化成numpy数组
            mask_pred = net(image)  # 1*15*256*256 float32

            # 预测结果采用one-hot编码,为每一个可能的类创建一个输出通道。通过取每个像素点在各个channel的argmax可以得到最终的预测分割图
            pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float().cpu().numpy() # 1*15*256*256 float32
            # pred = mask_pred.argmax(dim=1).cpu().numpy()  # 1*256*256 float32

            # 读取一张对应的标签，转化成numpy数组
            label = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float().cpu().numpy()  # 1*256*256—>1*15*256*256  float32
            # label = mask_true.float().cpu().numpy()  # 1*256*256  float32

            # 对一张图片计算15×15的hist矩阵，并累加
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

            # 每计算30张就输出一下目前已计算的图片中所有类别平均的mIoU值
            n += 1
            # if n > 0 and n % 30 == 0:
            #     print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(n, len(dataloader),
            #                                                           100 * np.nanmean(per_class_iu(hist)),
            #                                                           100 * np.nanmean(per_class_PA(hist))))

    #  计算所有验证集图片的逐类别mIoU值
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)

    # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> mIoU: ' + str(np.nanmean(mIoUs) * 100) + '; mPA: ' + str(np.nanmean(mPA) * 100))
    return np.nanmean(mIoUs) * 100
