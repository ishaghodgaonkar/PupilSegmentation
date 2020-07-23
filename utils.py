# Data Loader
# Dataset credits: https://github.com/Gyoorey/PupilDataset
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class PupilDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.iter = iter(os.listdir(root_dir + '/data'))
        self.iter = iter(os.listdir(root_dir + '/mask'))
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(os.listdir(root_dir + '/data'))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        img_path = next(self.iter)
        mask_path = next(self.iter)
        img = self.transform(Image.open(self.root_dir + '/data/' + img_path))
        mask = self.transform(Image.open(self.root_dir + '/mask/' + mask_path))
        while (img.shape != torch.Size([1, 256, 256]) or mask.shape != torch.Size([1, 256, 256])):
            img_path = next(self.iter)
            mask_path = next(self.iter)
            img = self.transform(Image.open(self.root_dir + '/data/' + img_path))
            mask = self.transform(Image.open(self.root_dir + '/mask/' + mask_path))

        return img, mask
