import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np

import preprocess


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return np.load(path)


class myImageFolder(data.Dataset):
    def __init__(self,
                 equi_infos,
                 up,
                 down,
                 up_disparity,
                 training,
                 loader=default_loader,
                 dploader=disparity_loader):

        self.up = up
        self.down = down
        self.disp_name = up_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.equi_infos = equi_infos

    def __getitem__(self, index):
        up = self.up[index]
        down = self.down[index]
        disp_name = self.disp_name[index]
        equi_info = self.equi_infos

        up_img = self.loader(up)
        down_img = self.loader(down)
        disp = self.dploader(disp_name)
        up_img = np.concatenate([np.array(up_img), equi_info], 2)
        down_img = np.concatenate([np.array(down_img), equi_info], 2)

        if self.training:
            h, w = up_img.shape[0], up_img.shape[1]
            th, tw = 512, 256

            # vertical remaining cropping
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            up_img = up_img[y1:y1 + th, x1:x1 + tw, :]
            down_img = down_img[y1:y1 + th, x1:x1 + tw, :]
            disp = np.ascontiguousarray(disp, dtype=np.float32)
            disp = disp[y1:y1 + th, x1:x1 + tw]

            # preprocessing
            processed = preprocess.get_transform(augment=False)
            up_img = processed(up_img)
            down_img = processed(down_img)

            return up_img, down_img, disp
        else:
            disp = np.ascontiguousarray(disp, dtype=np.float32)

            processed = preprocess.get_transform(augment=False)
            up_img = processed(up_img)
            down_img = processed(down_img)

            return up_img, down_img, disp

    def __len__(self):
        return len(self.up)
