import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from os.path import join

from .base_dataset import BaseDataset


class ColorDataset(BaseDataset):
    def __init__(self, opt, **kwargs):
        BaseDataset.__init__(self, opt)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if type(self.root) == str:
            self.data, self.length = self.parse_path(self.root, self.phase)
        elif type(self.root) == dict:
            # currently assume interface passes "processed" keypoints and masks, might need refactor
            self.data = self.process_dict(self.root)
            self.length = self.get_length_from_dict()
        else:
            raise TypeError("Expect type of root to be str or dict, but get: ", type(self.root))

        if self.length == 0:
            raise RuntimeError("Found 0 samples in root.")

    def __getitem__(self, index):
        data = {}
        data['latents'] = self.data['latents'][index]
        data['colors'] = self.data['colors'][index]
        data['color_masks'] = self.data['color_masks'][index]
        data['background_masks'] = self.data['background_masks'][index]
        return data

    def __len__(self):
        return self.length

    def parse_path(self, root, phase):
        """Given path to the dataroot, output a dictionary of data."""
        data_dir = join(root, phase)
        with open(join(data_dir, 'counter'), 'r') as f:
            t = int(f.read())

        if self.phase == 'train' and self.max_train_samples is not None:
            assert self.max_train_samples <= t, f'--max_train_samples should not be larger than the dataset size {t}, but got {self.max_train_samples}'
            t = self.max_train_samples

        data = {'latents': [], 'background_masks': [], 'color_masks': [], 'colors': [], 'targets': []}
        latent_dir = join(data_dir, 'latents')
        mask_dir = join(data_dir, 'masks')
        target_dir = join(data_dir, 'targets')

        for i in range(t):
            # latent shape is expected to be [1, G.mapping.num_ws, G.w_dim]
            latent = torch.load(join(latent_dir, f'{i}_w.pth'))
            data['latents'].append(latent[0])

            mask_data = np.load(join(mask_dir, f'{i}.npz'))
            color = torch.from_numpy(mask_data['color'] / 255 * 2 - 1).float().permute(2, 0, 1)
            color_mask = torch.from_numpy(mask_data['color_mask']).float()
            background_mask = torch.from_numpy(mask_data['background_mask']).float()
            data['colors'].append(color)
            data['color_masks'].append(color_mask)
            data['background_masks'].append(background_mask)

            target = Image.open(join(target_dir, f'{i}.png')).convert('RGB')
            target = self.transform(target)
            data['targets'].append(target)

        return data, t

    def process_dict(d):
        # TODO
        return d

    def get_length_from_dict(self):
        t = len(self.data['latents'])
        for v in self.data.values():
            assert t == len(v), "Length of each list in self.data should be equal."
        return t
