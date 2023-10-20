import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        # self.file_name = os.listdir(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w = pil_img.shape[1]
        h = pil_img.shape[0]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        img_ndarray = cv2.resize(pil_img, (newW, newH))
        # img_ndarray = cv2.resize(img_ndarray, (256, 256))
        img_ndarray = cv2.resize(img_ndarray, (512, 512))

        # 语义分割的image处理
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load_mask(cls, filename):
        # ext = splitext(filename)[1]
        # if ext in ['.npz', '.npy']:
        #     return Image.fromarray(np.load(filename))
        # elif ext in ['.pt', '.pth']:
        #     return Image.fromarray(torch.load(filename).numpy())
        # else:
        return np.asarray(Image.open(filename))



    def load_image(cls, filename):
        # ext = splitext(filename)[1]
        # if ext in ['.npz', '.npy']:
        #     return Image.fromarray(np.load(filename))
        # elif ext in ['.pt', '.pth']:
        #     return Image.fromarray(torch.load(filename).numpy())
        # else:
        return cv2.imread(str(filename))



    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # mask = self.load_mask(mask_file[0])

        # 语义分割Semantic_masks图像处理
        mask = self.load_image(mask_file[0])

        # 当语义分割的标注图像是RGB格式的，需要转为Gray格式的mask
        if mask.ndim == 3:
            # 1. 转为灰度图
            gray_img = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

            # 2. 找出颜色种类
            uniqueRs, index, count = np.unique(gray_img, return_inverse=True, return_counts=True)

            # 3. 构建dict
            mapDict = {}
            # 这里一开始是默认每个颜色按顺序对应index
            for i in range(len(uniqueRs)):
                mapDict[uniqueRs[i]] = i

            # 4. 映射,并将数组类型改为uint8
            mask = np.array([mapDict[x] for x in uniqueRs])[index].reshape(gray_img.shape).astype(np.uint8)
            # cv2.imwrite(savePath, mask)

        img = self.load_image(img_file[0])

        # assert img.size == mask.size, \
        #     'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        # super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_instance_color_RGB')
