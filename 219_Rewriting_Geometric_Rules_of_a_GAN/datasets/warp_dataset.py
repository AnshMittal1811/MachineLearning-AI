import torch
import torchvision.transforms as transforms
from PIL import Image
from os.path import join

from .base_dataset import BaseDataset
from util.util import load_mask, load_keypoints, get_warp_grid_normalized


class WarpDataset(BaseDataset):
    def __init__(self, opt, **kwargs):
        BaseDataset.__init__(self, opt)
        self.image_res = opt.image_res
        self.target_res = opt.target_res
        self.warp_cropped_car = opt.warp_cropped_car

        self.transform = transforms.Compose([
            transforms.Resize(self.image_res),
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
        data['warp_grid'] = self.data['warp_grid'][index]
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

        data = {'latents': [], 'keypoints': [], 'warp_grid': [], 'targets': []}
        latent_dir = join(data_dir, 'latents')
        keypoint_dir = join(data_dir, 'keypoints')
        target_dir = join(data_dir, 'targets')

        # only val and test split have user-defined masks for changed regions ('target_masks')
        if phase != 'train':
            data['target_masks'] = []
            target_mask_dir = join(data_dir, 'target_masks')
            target_mask_sz = (self.image_res, self.image_res)

        for i in range(t):
            # latent shape is expected to be [1, G.mapping.num_ws, G.w_dim]
            latent = torch.load(join(latent_dir, f'{i}_w.pth'))
            data['latents'].append(latent[0])

            w = self.image_res
            h = self.image_res * 3 // 4 if self.warp_cropped_car else self.image_res
            keypoints = load_keypoints(join(keypoint_dir, f'{i}.npy'), w, h)
            data['keypoints'].append(keypoints)

            nw = self.target_res
            nh = self.target_res * 3 // 4 if self.warp_cropped_car else self.target_res
            grid = get_warp_grid_normalized(keypoints, (h, w), (nh, nw))
            data['warp_grid'].append(torch.from_numpy(grid))

            target = Image.open(join(target_dir, f'{i}.png')).convert('RGB')
            target = self.transform(target)
            data['targets'].append(target)

            if phase != 'train':
                target_mask = load_mask(join(target_mask_dir, f'{i}.png'), size=target_mask_sz)
                target_mask = torch.from_numpy(target_mask)
                data['target_masks'].append(target_mask)

        return data, t

    def process_dict(d):
        # TODO
        return d

    def get_length_from_dict(self):
        t = len(self.data['latents'])
        for v in self.data.values():
            assert t == len(v), "Length of each list in self.data should be equal."
        return t
