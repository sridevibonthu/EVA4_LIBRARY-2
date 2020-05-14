import torch
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import random
import re


def validatePath(path):
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valild path")


def fgbg_test_train(folder, train=0.8, limit=1):
    # we have fixed file names from image000001 to image 400000, same for depth and mask
    # so we can easily shuffle and load
    validatePath(folder)
    images_path = os.path.join(folder, 'images')
    depths_path = os.path.join(folder, 'depth')
    masks_path = os.path.join(folder, 'masks')
    validatePath(images_path)
    validatePath(depths_path)
    validatePath(masks_path)

    # we are expecting the files in a particular format
    images = sorted(filter(lambda x: re.match(r'^fgbg[\d]{6}\.jpg$', x), os.listdir(images_path)))
    depths = sorted(filter(lambda x: re.match(r'^fgbg[\d]{6}\.jpg$', x), os.listdir(depths_path)))
    masks = sorted(filter(lambda x: re.match(r'^mask[\d]{6}\.jpg$', x), os.listdir(masks_path)))

    images = list(map(lambda x: os.path.join(images_path, x), images))
    depths = list(map(lambda x: os.path.join(depths_path, x), depths))
    masks = list(map(lambda x: os.path.join(masks_path, x), masks))


    if(len(images) != len(depths) != len(masks)):
        raise ValueError("image counts do not match in images, depth and masks")

    l = len(images)
    if limit<1:
        l = int(len(images)*limit)

    dataset = list(zip(images, masks, depths))
    random.shuffle(dataset)

    ts = int(l * train)
    return dataset[:ts], dataset[ts:l]


class FGBGDataset(Dataset):
    """Tine Imagenet dataset reader."""

    def __init__(self, data, transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.transform = transform
        self.images, self.masks, self.depths = zip(*data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx], as_gray=False, pilmode="RGB")
        mask = io.imread(self.masks[idx], as_gray=True, pilmode="1")
        depth = io.imread(self.depths[idx], as_gray=True, pilmode="L")

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask/255)
        depth = torch.from_numpy(depth/255)
        # Scale mask and depth to 0-1 range
        # we need not normalize our outputs
        return image, torch.stack([mask, depth])
