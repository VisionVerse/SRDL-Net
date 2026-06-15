import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        img_A = np.array(img_A).astype(np.float32)
        img_B = np.array(img_B).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img_A /= 255.0
        img_A -= self.mean
        img_A /= self.std

        img_B /= 255.0
        img_B -= self.mean
        img_B /= self.std

        return {'A': img_A, 'B': img_B, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        img_B = np.array(img_B).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img_A = torch.from_numpy(img_A).float()
        img_B = torch.from_numpy(img_B).float()
        mask = torch.from_numpy(mask).float()

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        if random.random() < 0.5:
            img_A = img_A.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img_B = img_B.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'A': img_A, 'B': img_B, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        w, h = img_A.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img_A = img_A.resize((ow, oh), Image.BILINEAR)
        img_B = img_B.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img_A.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img_A = img_A.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_B = img_B.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'A': img_A, 'B': img_B, 'label': mask}
