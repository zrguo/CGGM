from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import random
import torch


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size=(160, 160)):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size=(160, 160)):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 3)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis - 1).copy()

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return image


def cutout(image, label, patch_size=(30, 30)):
    mod, image_height, image_width = image.shape
    patch_height, patch_width = patch_size

    x = np.random.randint(0, image_height - patch_height)
    y = np.random.randint(0, image_width - patch_width)

    for i in range(image.shape[0]):
        modality = image[i, :, :]
        modality[x:x + patch_height, y:y + patch_width] = 0
        image[i, :, :] = modality

    label[x:x + patch_height, y:y + patch_width] = 0

    return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label
    

class BraTSData(Dataset):
    def __init__(self, root, mode, size=(160, 160)):
        self.root = root
        self.mode = mode
        self.size = size
        self.root = os.path.join(self.root, self.mode)
        data = os.listdir(self.root)
        self.data = [os.path.join(self.root, d) for d in data]

    def __getitem__(self, item):
        id = self.data[item]
        h5f = h5py.File(id, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label == 4] = 3
        idx = np.random.randint(0, 128)
        while np.max(label[:, :, idx]) == 0:
            idx = np.random.randint(0, 128)
        image = image[:, :, :, idx]
        label = label[:, :, idx]

        if self.mode == 'train':
            sample = {'image': image, 'label': label}
            sample = CenterCrop(self.size)(sample)
            sample = RandomRotFlip()(sample)
            image, label = sample['image'], sample['label']
            image = GaussianNoise(p=0.1)(image)
            sample = {'image': image, 'label': label}
            image, label = ToTensor()(sample)
        else:
            sample = {'image': image, 'label': label}
            sample = CenterCrop(self.size)(sample)
            image, label = ToTensor()(sample)

        flair, t1ce, t1, t2 = image[0].unsqueeze(dim=0), image[1].unsqueeze(dim=0), image[2].unsqueeze(dim=0), image[3].unsqueeze(dim=0)

        return flair, t1ce, t1, t2, label

    def __len__(self):
        return len(self.data)



