# Based on SFNet

import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(dataroot, batch_size=64, num_workers=0, task_name='raincityscape', use_transform=True, patch_size=128):
    crop_size = patch_size
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(crop_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(dataroot, task_name=task_name, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(dataroot, task_name='raincityscape', batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(dataroot, task_name, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(dataroot, task_name='raincityscape', batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(dataroot, task_name, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, dataroot, task_name=None, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test
        TXT_file = os.path.join(dataroot, task_name, 'TRAIN.txt' if not is_test else 'TEST.txt')
        self.inp_filenames = []
        self.tar_filenames = []
        with open(TXT_file, 'r') as f:
            for line in f:
                inp_path, tar_path = line.strip().split()
                self.inp_filenames.append(inp_path)
                self.tar_filenames.append(tar_path)

    def __len__(self):
        return len(self.inp_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.inp_filenames[idx]).convert('RGB')
        label = Image.open(self.tar_filenames[idx]).convert('RGB')

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = os.path.basename(self.inp_filenames[idx])
            return image, label, name
        return image, label

