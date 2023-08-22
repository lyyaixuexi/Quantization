import torch
import argparse
import os
import logging
from torch.utils import data
from dataloader_mulit_patch import img_cls_by_dir_loader as data_loader
from lr_scheduling import *
from tqdm import tqdm

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def cls_loss(pred, target, gpu_id):
    pred = pred[:, :, 1, 1].squeeze()
    criterion_cls = torch.nn.CrossEntropyLoss(reduce=True, size_average=True).cuda(
        gpu_id
    )
    loss_cls = criterion_cls(pred, target.long())
    return loss_cls


def cls_multi_patch_loss(pred, target):
    '''
    This loss is for the mutli patch concat input
    '''
    batchsize, c_num, _, _ = pred.shape
    pred = pred.permute(0, 2, 3, 1)
    pred = pred.reshape(-1, c_num)
    # pred = pred.permute(0, 2, 1)
    # pred = pred.reshape(-1)
    target = target.reshape(-1)
    loss_cls = torch.nn.functional.cross_entropy(pred, target.long())
    return loss_cls


def train(args, gpu_id=0):
    # Setup Dataloader
    with open('./train_list_test.txt','r') as f:
        lines = f.readlines()
        lines = [i.strip('\n') for i in lines]

    root_dir = lines
    if args.project == "HM":   # 
        args.multi_patch = False
        args.img_size = 80
        args.color_mode = 'yuv420sp'
    elif args.project == 't1q':
        args.multi_patch = True
        args.img_size = 288
        args.color_mode = 'YUV_bt601V'
    elif args.project in ['h1z', 'vgg', 'zerovgg']:
        args.multi_patch = False
        args.img_size = 96
        args.color_mode = 'bgr'

    dataset = data_loader(root_dir, split="train", is_transform=True, img_size=args.img_size, color_mode=args.color_mode, multi_patch = args.multi_patch)

    trainloader = data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,pin_memory=True
    )

    ## Setup Model
    if args.project in ["t1q", 'h1z']:
        from models.traffic_sign_cls_1 import traffic_sign_cls as ClassifyNet
    elif args.project == 'HM':
        from model.traffic_sign_cls_HM import traffic_sign_cls as ClassifyNet
    elif args.project == 'vgg':
        from model.vgg import My_VGG16 as ClassifyNet
    elif args.project == 'zerovgg':
        from model.zero_vgg import Zero_VGG16 as ClassifyNet

    if args.project in ['vgg', 'zerovgg']:
        model = ClassifyNet().cuda(gpu_id)
    else:
        model = ClassifyNet(c1=3, n_classes=args.n_classes, color_mode=args.color_mode).cuda(gpu_id)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.l_rate,
        weight_decay=5e-4,
        momentum=0.9,
    )

    ## Training Part
    ##-----epoch-----##
    for epoch in range(0, args.n_epoch + 1):
        model.train()
        adjust_learning_rate(optimizer, args.l_rate, epoch)
        ##-----iteration-----#
        pbar = tqdm(range(len(trainloader)))
        for i, (images, lbls) in zip(pbar, trainloader):
            images = images.cuda(gpu_id)
            lbls = lbls.cuda(gpu_id)
            optimizer.zero_grad()
            if args.project == 'HM':
                images = list(torch.split(images, [80, 40], dim=2))
                pred = model(*images)
            else:
                pred = model(images)

            loss = cls_multi_patch_loss(pred, lbls)
            loss.backward()
            optimizer.step()

            ##----logger info-----##
            if (i + 1) % (len(trainloader)%25) == 0:
                logger.info(
                    "Epoch [%d/%d-%.4f%%]  Loss: %.4f  lr = %f"
                    % (
                        epoch + 1,
                        args.n_epoch,
                        i / len(trainloader) * 100,
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        ##-----valuation-----##
        '''
        这边希望整个valuation是绝对独立的一个整体，会独立做数据初始化和网络加载
        '''
        # if args.valuation and epoch % 2 == 0:
        #     ## TODO
        #     pass
            # with torch.no_grad():
            #     validate(a,model)

        ##-----save------##
        if epoch % 1 == 0:
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(
                model.cpu().state_dict(),
                "{}/{}.pkl".format(save_path, epoch),   #{}_{}_ args.arch, args.dataset, 
            )
            model.cuda(gpu_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--n_classes", nargs="?", type=int, default=279, help=""
        )
    parser.add_argument(
        "--n_epoch", nargs="?", type=int, default=25, help="# of the epochs"
        )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=512, help="Batch Size"    # 1024   vgg=512 zerovgg=256
        )
    parser.add_argument(
        "--l_rate", nargs="?", type=float, default=1e-2, help="Learning Rate"
        )
    parser.add_argument(
        "--project", nargs="?", type=str, default="t1q", help="negative patch folder",   # h1z t1q HM
        )
    # parser.add_argument(
    #     "--img_size", nargs="?", type=str, default=288, help="negative patch folder",   # h1z=96  HM=80  #t1q=288
    #     )
    # parser.add_argument(
    #     "--color_mode", nargs="?", type=str, default="yuv420sp",   # h1z=bgr t1q=YUV_bt601V HM=yuv420sp
    #     )
    parser.add_argument(
        "--save_path",
        nargs="?",
        type=str,
        default="./pytorch_models/tsr_t1q_YUV_bt601V_20230518",    # tsr_h1z_bgr_20230216  tsr_HM_YUV420sp_20230212 tsr_t1q_YUV_bt601V_20230228
        )

    args = parser.parse_args()

    train(args, gpu_id=1)
