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

def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

class FGBGDataset(Dataset):
    """Tine Imagenet dataset reader."""

    def __init__(self, data, quantize=None, image_transform=None, mask_transform=None, depth_transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.depth_transform = depth_transform
        self.images, self.masks, self.depths = zip(*data)
        self.quantize = quantize
        if quantize and len(quantize) != 2:
            raise ValueError("quatize must be a a tuple of two integers")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx], as_gray=False, pilmode="RGB")
        # for grayscale images tensor will need to transpose it.
        mask = io.imread(self.masks[idx], as_gray=True, pilmode="1").T 
        depth = io.imread(self.depths[idx], as_gray=True, pilmode="L").T

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.depth_transform:
            depth = self.depth_transform(depth)

        #mask = torch.from_numpy(mask/255)
        #depth = torch.from_numpy(depth/255)
        # Scale mask and depth to 0-1 range
        # we need not normalize our outputs
        if not self.quantize:
            return image, torch.stack([mask, depth])
        
        m = (mask*quantize[0]).int()
        
        d = (depth*quantize[1]).int()
        d = make_one_hot(d, quantize[1])[1:, : ,:]

            
