import torch
import cv2
import os
import random
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class z_normalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image.astype(np.float32)
        image_mean = np.mean(image)
        image_std = np.std(image)
        image_norm = (image - image_mean) / image_std
        return {'image': image_norm, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        label = sample['label']
        label = np.ascontiguousarray(label)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}


class Normalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image.astype(np.float32) / 255.
        return {'image': image, 'label': label}


class Random_Crop(object):
    def __init__(self, crop_size, img_size):
        self.crop_size = crop_size
        self.img_size = img_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        h = self.crop_size[0]
        w = self.crop_size[1]

        H = random.randint(0, self.img_size[0] - h)
        W = random.randint(0, self.img_size[1] - w)

        image = image[H: H + h, W: W + w, :]
        label = label[H: H + h, W: W + w]

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.randint(0, 1) > 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        elif random.randint(0, 1) > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return {'image': image, 'label': label}


class Random_Rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.randint(0, 1) > 0.5:
            image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
            label = cv2.rotate(label, cv2.cv2.ROTATE_90_CLOCKWISE)
        return {'image': image, 'label': label}


class Scaling(object):
    def __init__(self, scale_size):
        self.scale_size = scale_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = cv2.resize(image, self.scale_size)
        label = cv2.resize(label, self.scale_size)
        return {'image': image, 'label': label}


def transform(sample, crop_size, scale_size):
    trans = transforms.Compose([
        Scaling(scale_size),
        Random_Crop(crop_size, scale_size),
        Random_Flip(),
        Random_Rotate(),
        Normalization(),
        ToTensor()
    ])
    return trans(sample)


def transform_valid(sample, crop_size):
    trans = transforms.Compose([
        Scaling(crop_size),
        Normalization(),
        ToTensor()
    ])
    return trans(sample)

class dataset(Dataset):
    def __init__(self, list_file_images, scale_size=(1000, 1000), crop_size=(900, 900), root_img='', root_label='',
                 mode='train'):
        self.lines = []
        self.lines_label = []
        self.paths_images = []
        self.paths_labels = []
        self.mode = mode
        self.scale_size = scale_size
        self.crop_size = crop_size
        with open(list_file_images, encoding='utf-8-sig') as f:
            for line in f:
                path_image = os.path.join(root_img, line)
                line_label = line[0:-5] + '_segmentation' + '.png'
                path_label = os.path.join(root_label, line_label)
                self.paths_images.append(path_image)
                self.paths_labels.append(path_label)
                self.lines.append(line)
                self.lines_label.append(line_label)

    def __getitem__(self, item):
        path_image = self.paths_images[item]
        path_image = path_image.rstrip("\n")
        path_label = self.paths_labels[item]
        path_label = path_label.rstrip("\n")

        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(path_label, 0)
        sample = {'image': image, 'label': label}
        if self.mode == 'train':
            sample = transform(sample, self.crop_size, self.scale_size)
        else:
            sample = transform_valid(sample, self.scale_size)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.lines)
