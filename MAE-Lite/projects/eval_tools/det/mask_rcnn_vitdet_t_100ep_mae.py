from .mask_rcnn_vitdet_t_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "checkpoints/mae_tiny_400e.pth?matching_heuristics=True"
train.output_dir = "output/ViTDet/mask_rcnn_vitdet_t_100ep_mae"