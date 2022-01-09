import numpy as np
import os
import torch
import pickle
import random
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class z_normalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image=image.astype(np.float32)
        image_mean=np.mean(image)
        image_std=np.std(image)
        image_norm=(image-image_mean)/image_std
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
  def __call__(self,sample):
    image=sample['image']
    label=sample['label']
    image=image.astype(np.float32)/255.
    return {'image':image, 'label':label}

class Random_Crop(object):
    def __init__(self,crop_size,img_size):
      self.crop_size=crop_size
      self.img_size=img_size
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        h = self.crop_size[0]
        w = self.crop_size[1]
    
        H = random.randint(0, self.img_size[0] - h)
        W = random.randint(0, self.img_size[1] - w)

        image = image[H: H + h, W: W + w,:]
        label = label[H: H + h, W: W + w]

        return {'image': image, 'label': label}

def transform(sample,crop_size):
    trans = transforms.Compose([
        Random_Crop(crop_size),
        z_normalization(),
        ToTensor()
    ])
    return trans(sample)

class dataset(Dataset):
    def __init__(self, list_file_images,crop_size, root='', mode='train'):
        self.lines = []
        self.paths_images=[]
        self.paths_labels=[]
        self.mode = mode
        self.crop_size=crop_size
        with open(list_file_images, encoding='utf-8-sig') as f:
            for line in f:
                path_image = os.path.join(root, 'image_2')
                path_image = os.path.join(path_image, line)
                path_label = os.path.join(root, 'semantic')
                path_label = os.path.join(path_label, line)
                self.paths_images.append(path_image)
                self.paths_labels.append(path_label)
                self.lines.append(line)

    def __getitem__(self, item):
        path_image = self.paths_images[item]
        path_image = path_image.rstrip("\n")
        path_label = self.paths_labels[item]
        path_label = path_label.rstrip("\n")
        
        image=cv2.imread(path_image)
        label=cv2.imread(path_label,0)
        sample={'image':image,'label':label}
        sample=transform(sample,self.crop_size,image.shape)
        return sample['image'],sample['label']


    def __len__(self):
        return len(self.names)
