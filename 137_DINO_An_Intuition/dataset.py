import torch
from torch.utils.data import Dataset
import os
import glob
import imageio


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imageio.imread(self.image_paths[idx])
        if self.transform is None: # validation
            return torch.tensor(image)
        crops = self.transform(image)
        return crops