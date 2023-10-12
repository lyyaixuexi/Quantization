import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', mode='fine', augment=False):
        self.root=os.path.expanduser(root)
        self.mode='gtFine' if mode=='fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit',split)
        self.targets_dir = os.path.join(self.root, self.mode, split)

        print("images_dir:{}".format(self.images_dir))
        print("targets_dir:{}".format(self.targets_dir))

        self.split = split
        self.augment=augment
        self.images=[]
        self.targets=[]
        self.mapping={
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 2,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 3,  # building
            12: 4,  # wall
            13: 5,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 6,  # pole
            18: 0,  # polegroup
            19: 7,  # traffic light
            20: 8,  # traffic sign
            21: 9,  # vegetation
            22: 10,  # terrain
            23: 11,  # sky
            24: 12,  # person
            25: 13,  # rider
            26: 14,  # car
            27: 15,  # truck
            28: 16,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 17,  # train
            32: 18,  # motorcycle
            33: 19,  # bicycle
            -1: 0  # licenseplate
        }
        self.mappingrgb={
            0: (0, 0, 0),  # unlabeled
            1: (0, 0, 0),  # ego vehicle
            2: (0, 0, 0),  # rect border
            3: (0, 0, 0),  # out of roi
            4: (0, 0, 0),  # static
            5: (111, 74,  0),  # dynamic
            6: (81,  0, 81),  # ground
            7: (128, 64,128),  # road
            8: (244, 35,232),  # sidewalk
            9: (250,170,160),  # parking
            10: (230,150,140),  # rail track
            11: (70, 70, 70),  # building
            12: (102,102,156),  # wall
            13: (190,153,153),  # fence
            14: (180,165,180),  # guard rail
            15: (150,100,100),  # bridge
            16: (150,120, 90),  # tunnel
            17: (153,153,153),  # pole
            18: (153,153,153),  # polegroup
            19: (250, 170, 30),  # traffic light
            20: (220, 220,  0),  # traffic sign
            21: (107, 142, 35),  # vegetation
            22: (152, 251,152),  # terrain
            23: (70,130,180),  # sky
            24: (220, 20, 60),  # person
            25: (255, 0,  0),  # rider
            26: (0, 0,142),  # car
            27: (0, 0, 70),  # truck
            28: (0, 60, 100),  # bus
            29: (0, 0, 90),  # caravan
            30: (0, 0, 110),  # trailer
            31: (0, 80, 100),  # train
            32: (0, 0, 230),  # motorcycle
            33: (119, 11, 32),  # bicycle
            -1: (0,  0, 142)  # licenseplate
        }

        self.num_classes=20

        if mode not in ['fine', 'coarse']:
            raise ValueError('Invalid mode, please use mode="fine" or "coarse"')
        if mode=='fine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "fine", please use split="train" or "test" or "val"')
        elif mode=='coarse' and split not in ['train', 'train_extra', 'val']:
            raise ValueError('Invalid split for mode "coarse", please use split="train" or "train_extra" or "val"')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name='{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                self.targets.append(os.path.join(target_dir, target_name))

    def __len__(self):
            return len(self.images)

    def mask_to_class(self, mask):
        assert mask.dim()==2
        masking=torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            masking[mask==k]=self.mapping[k]
        return masking

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('L')

        if self.augment:
            image = TF.resize(image, size=(256+40,512+40), interpolation=TF.InterpolationMode.NEAREST)
            target = TF.resize(target, size=(256+40,512+40), interpolation=TF.InterpolationMode.NEAREST)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256,512))
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)

        else:
            image = TF.resize(image, size=(256,512), interpolation=TF.InterpolationMode.NEAREST)
            # target = TF.resize(target, size=(512,1024), interpolation=TF.InterpolationMode.NEAREST)

        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        image=TF.to_tensor(image)
        targetmask = self.mask_to_class(target).long()

        return image, targetmask

if __name__ == '__main__':
    cityscapes=CityscapesDataset('~/UNet/data')
    print(cityscapes.__getitem__(0))

