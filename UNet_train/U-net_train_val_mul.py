import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate, compute_mIoU
from unet import UNet
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

img_scale = 0.5
batch_size = 16
finetune_epoch = 20
learning_rate = 1e-5
dir_checkpoint = Path('./checkpoints/ISAID_512')
save_checkpoint = True
bilinear = True
amp = False

def finetune(train_loader, val_loader, model, finetune_epoch):
    for epoch in range(finetune_epoch):

        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{finetune_epoch}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            val_score = evaluate(model, val_loader, device)
            logging.info('Validation Dice score: {}'.format(val_score))
            scheduler.step(val_score)

            mIoU = compute_mIoU(model, val_loader, device)
            logging.info('Validation mIoU: {}'.format(mIoU))

            logging.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint and dist.get_rank() == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    return mIoU

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1)
    FLAGS = parser.parse_args()
    local_rank = int(FLAGS.local_rank)

    torch.cuda.set_device(int(local_rank))
    dist.init_process_group(backend='nccl')

    device = torch.device("cuda", int(local_rank))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Create dataset ISAID
    dir_img_train = Path('/home/vit/data/ISAID/train/image/images')
    dir_mask_train = Path('/home/vit/data/ISAID/train/Semantic_masks/images')
    dir_img_eval = Path('/home/vit/data/ISAID/val/val_images')
    dir_mask_eval = Path('/home/vit/data/ISAID/val/Semantic_masks/images')

    try:
        dataset_train = CarvanaDataset(dir_img_train, dir_mask_train, img_scale)
    except (AssertionError, RuntimeError):
        dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale)

    try:
        dataset_eval = CarvanaDataset(dir_img_eval, dir_mask_eval, img_scale)
    except (AssertionError, RuntimeError):
        dataset_eval = BasicDataset(dir_img_eval, dir_mask_eval, img_scale)

    # 2. Create data loaders
    n_train = len(dataset_train)
    n_val = len(dataset_eval)
    train_set = Subset(dataset_train, range(len(dataset_train)))
    val_set = Subset(dataset_eval, range(len(dataset_eval)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset=train_set, **loader_args, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_set, drop_last=True, **loader_args, sampler=val_sampler)

    logging.info(f'''Starting training:
            Epochs:          {finetune_epoch}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Images scaling:  {img_scale}
        ''')

    # 3. Load model
    model = UNet(n_channels=3, n_classes=15, bilinear=bilinear).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    # 5. Begin training
    accuracy = finetune(train_loader, val_loader, model, finetune_epoch)