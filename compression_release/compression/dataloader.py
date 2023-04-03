import os
import numpy as np

import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# torchvision.set_image_backend("accimage")


class DistributedIndicesWrapper(torch.utils.data.Dataset):
    """
    Utility wrapper so that torch.utils.data.distributed.DistributedSampler can work with train test splits
    """

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # TODO: do the sampling here ?
        idx = self.indices[item]
        return self.dataset[idx]


def get_cifar_dataloader(
    dataset,
    batch_size,
    n_threads=4,
    data_path="/home/dataset/",
    distributed=False,
    logger=None,
):
    """
    Get dataloader for cifar10/cifar100
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    if dataset == "cifar10":
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == "cifar100":
        norm_mean = [0.50705882, 0.48666667, 0.44078431]
        norm_std = [0.26745098, 0.25568627, 0.27607843]
    data_root = os.path.join(data_path, "cifar")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )

    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root=data_root, train=False, transform=val_transform
        )
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_root, train=True, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR100(
            root=data_root, train=False, transform=val_transform
        )
    else:
        logger.info("invalid data set")
        assert False, "invalid data set"

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=n_threads,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=n_threads,
    )
    return train_loader, val_loader


def get_cifar_train_val_test_dataloader(
    dataset,
    batch_size,
    n_threads=4,
    data_path="/home/dataset/",
    distributed=False,
    logger=None,
):
    """
    Get dataloader for cifar10/cifar100
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    if dataset == "cifar10":
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == "cifar100":
        norm_mean = [0.50705882, 0.48666667, 0.44078431]
        norm_std = [0.26745098, 0.25568627, 0.27607843]
    data_root = os.path.join(data_path, "cifar")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )

    if "cifar100" in dataset:
        train_dataset = datasets.CIFAR100(
            root=data_root, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR100(
            root=data_root, train=False, transform=val_transform
        )
    elif "cifar10" in dataset:
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, transform=val_transform
        )
    else:
        logger.info("invalid data set")
        assert False, "invalid data set"

    num_train = len(train_dataset)
    indices = list(range(num_train))
    ratio = 0.8
    split = int(num_train * ratio)
    # split = int(25000)

    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            DistributedIndicesWrapper(train_dataset, indices[:split])
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            DistributedIndicesWrapper(train_dataset, indices[split:num_train])
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]
        )
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=n_threads,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=n_threads,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=n_threads,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        val_sampler,
        test_sampler,
    )


def get_imagenet_dataloader(
    dataset,
    batch_size,
    n_threads=4,
    data_path="/home/dataset/",
    transfroms_name="default",
    distributed=False,
    logger=None,
):
    """
    Get dataloader for imagenet
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info(
        "|===>Get datalaoder for {}, transfrom: {}".format(dataset, transfroms_name)
    )

    dataset_path = os.path.join(data_path, dataset)
    traindir = os.path.join(dataset_path, "train")
    valdir = os.path.join(dataset_path, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if transfroms_name in ["default"]:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif transfroms_name in ["mobilenet"]:
        crop_scale = 0.20

        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif transfroms_name in ["mobilenetv3"]:
        crop_scale = 0.08

        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                brightness=32. / 255., saturation=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif transfroms_name in ["mobilenet_nas"]:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        raise NotImplementedError(
            "Data transform {} is not yet implemented.".format(transfroms_name)
        )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(traindir, train_transforms)
    val_dataset = datasets.ImageFolder(valdir, val_transforms)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=n_threads,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=n_threads,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_imagenet_train_val_test_dataloader(
    dataset,
    batch_size,
    n_threads=4,
    data_path="/home/dataset/",
    transfroms_name="default",
    distributed=False,
    logger=None,
):
    """
    Get dataloader for imagenet
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info(
        "|===>Get datalaoder for {}, transfrom: {}".format(dataset, transfroms_name)
    )

    dataset_path = os.path.join(data_path, dataset)
    traindir = os.path.join(dataset_path, "train")
    testdir = os.path.join(dataset_path, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if transfroms_name in ["default"]:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif transfroms_name in ["mobilenet"]:
        crop_scale = 0.20

        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif transfroms_name in ["mobilenet_nas"]:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        raise NotImplementedError(
            "Data transform {} is not yet implemented.".format(transfroms_name)
        )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(traindir, train_transforms)
    test_dataset = datasets.ImageFolder(testdir, test_transforms)

    g = torch.Generator()
    g.manual_seed(0)
    indices = torch.randperm(len(train_dataset), generator=g).tolist()
    train_split = indices[:-50000]
    val_split = indices[-50000:]
    train_subdataset = torch.utils.data.SubSet(train_dataset, train_split)
    val_subdataset = torch.utils.data.SubSet(train_dataset, val_split)
    # ratio = 0.8
    # split = int(num_train * ratio)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_subdataset
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_subdataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.RandomSampler(val_subdataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_subdataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=n_threads,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subdataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=n_threads,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=n_threads,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        val_sampler,
        test_sampler,
    )

