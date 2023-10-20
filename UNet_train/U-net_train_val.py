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
from evaluate import evaluate
from unet_dice_score_miou import compute_mIoU
from unet import UNet

img_scale = 0.5
batch_size = 32
finetune_epoch = 50
learning_rate = 1e-4
dir_checkpoint = Path('./checkpoints/ISAID')
save_checkpoint = True
bilinear = True
amp = False

def finetune(train_loader, val_loader, model, finetune_epoch):
    for epoch in range(finetune_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{finetune_epoch}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                # print(images.shape)
                true_masks = batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                    # print(F.softmax(masks_pred, dim=1).float().shape)
                    # print(F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float().shape)
                    # print(masks_pred.shape)
                    # print(true_masks.shape)
                    loss = criterion(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float()[:, 1:, ...],
                                       F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()[:, 1:, ...],
                                       multiclass=True)
                    # loss = dice_loss(F.softmax(masks_pred, dim=1).float()[:, 1:, ...],
                    #                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()[:, 1:, ...],
                    #                    multiclass=True)

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

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint512_withCE_lr1e4_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    return mIoU

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Create dataset ISAID
    dir_img_train = Path('/home/liaoyuyan/dataset/ISAID/train/image/images')
    dir_mask_train = Path('/home/liaoyuyan/dataset/ISAID/train/Semantic_masks/images')
    dir_img_eval = Path('/home/liaoyuyan/dataset/ISAID/val/val_images')
    dir_mask_eval = Path('/home/liaoyuyan/dataset/ISAID/val/Semantic_masks/images')

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

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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
    model = UNet(n_channels=3, n_classes=15, bilinear=bilinear).cuda()

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    # 5. Begin training
    accuracy = finetune(train_loader, val_loader, model, finetune_epoch)