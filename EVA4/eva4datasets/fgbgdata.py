import torch
import os
from skimage import io
from skimage import feature
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

def combine_classes(labels, onehotclasses, device):
  l = (labels[:, :1, :, :]*onehotclasses[0]).long()
  v = make_one_hot(l, onehotclasses[0], device)[:, 1:, :, :]
  j = 1
  for i in range(1, len(onehotclasses)):
    l = (labels[:, j:j+1, :, :]*onehotclasses[i]).long()
    x = make_one_hot(l, onehotclasses[i], device)[:, 1:, :, :]
    j = j+1
    v = torch.cat((v,x), 1)

  return v

class FGBGDataset(Dataset):
    """Tine Imagenet dataset reader."""

    def __init__(self, data, image_transform=None, mask_transform=None, depth_transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.depth_transform = depth_transform
        self.images, self.masks, self.depths = zip(*data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx], as_gray=False, pilmode="RGB")
        # for grayscale images tensor will need to transpose it.
        mask = io.imread(self.masks[idx], as_gray=True, pilmode="1").T 
        depth = io.imread(self.depths[idx], as_gray=True, pilmode="L").T

        #maskedge = feature.canny(io.imread(self.masks[idx], as_gray=True, pilmode="1").T)
        #depthedge = feature.canny(io.imread(self.depths[idx], as_gray=True, pilmode="L").T, sigma = 0.5)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            #maskedge = self.mask_transform(maskedge)

        if self.depth_transform:
            depth = self.depth_transform(depth)
            #depthedge = self.depth_transform(depthedge)

        # get edge images for mask and depth for sharpness


        return image, torch.stack([mask, depth])
        #return image, torch.stack([mask, depth, maskedge, depthedge])
        

            
