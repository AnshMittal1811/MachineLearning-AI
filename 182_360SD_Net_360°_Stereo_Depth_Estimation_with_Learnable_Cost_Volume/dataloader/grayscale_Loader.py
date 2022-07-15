import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np


def default_loader(path):
    return np.tile(
        np.expand_dims(np.array(Image.open(path).convert('LA'))[:, :, 0], 2),
        (1, 1, 3))  # convert RGB image to grayscale and copy for 3 channels


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

            # grayscale preprocess
            trans = transforms.Compose([transforms.ToTensor()])
            up_img = trans(up_img)
            down_img = trans(down_img)

            return up_img, down_img, disp
        else:
            disp = np.ascontiguousarray(disp, dtype=np.float32)

            # grayscale preprocess
            coord_trans = transforms.Compose([transforms.ToTensor()])
            up_img = coord_trans(up_img)
            down_img = coord_trans(down_img)

            return up_img, down_img, disp

    def __len__(self):
        return len(self.up)
