from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import scipy.io as scio
import h5py
import math
import glob
import os

class DatasetConstructor(data.Dataset):
    def __init__(self):
        self.datasets_com = ['SHA', 'SHB', 'QNRF_small', 'OUC', 'GCC'] # add NW
        return

    # return current dataset(SHA/SHB/QNRF) for current image
    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]

    def resize(self, img, dataset_name):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA":
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height
            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset_name == "SHB":
            resize_height = height
            resize_width = width
        elif dataset_name == 'GCC':
            resize_height = 544
            resize_width = 960
        elif dataset_name.find("OUC") != -1: # it is QNRF_large
            if resize_width >= 1024:
                tmp = resize_width
                resize_width = 1024
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 1024:
                tmp = resize_height
                resize_height = 1024
                resize_width = (resize_height / tmp) * resize_width

            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 1024/416: # 1024/416=2.46
                    resize_width = 1024
                    resize_height = 416
            else:
                if resize_height / resize_width > 1024/416:
                    resize_height = 1024
                    resize_width = 416

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32

        elif dataset_name.find("QNRF_small") != -1 or dataset_name.find("NWPU") != -1:
            if resize_width >= 1024:
                tmp = resize_width
                resize_width = 1024
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 1024:
                tmp = resize_height
                resize_height = 1024
                resize_width = (resize_height / tmp) * resize_width

            if resize_height <= 512:
                tmp = resize_height
                resize_height = 512
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 512:
                tmp = resize_width
                resize_width = 512
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 1024 / 512:
                    resize_width = 1024
                    resize_height = 512
            else:
                if resize_height / resize_width > 1024 / 512:
                    resize_height = 1024
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img


class TrainDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 mode='crop',
                 dataset_name="QNRF",
                 device=None,
                 is_random_hsi=False,
                 is_flip=False,
                 fine_size=400
                 ):
        super(TrainDatasetConstructor, self).__init__()
        self.train_num = train_num
        self.imgs = []
        self.fine_size = fine_size
        self.data_root, self.gt_root = data_dir_path, gt_dir_path
        self.mode = mode
        self.device = device
        self.is_random_hsi = is_random_hsi
        self.is_flip = is_flip
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs

        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.train_num = len(imgs)

        # changed here --------------------------
        print('Constructing training dataset...')
        for i in range(self.train_num):
            img_tmp = imgs[i]
            den = os.path.join(self.gt_root, os.path.splitext(os.path.basename(img_tmp))[0] + ".npy")
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den])


    def __getitem__(self, index):
        if self.mode == 'crop':

            img_path, gt_map_path = self.imgs[index]

            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(TrainDatasetConstructor, self).get_cur_dataset(img_path)
            img = super(TrainDatasetConstructor, self).resize(img, cur_dataset)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))

            if self.is_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)

            img, gt_map = transforms.ToTensor()(np.array(img)), transforms.ToTensor()(np.array(gt_map))
            img_shape = img.shape  # C, H, W
            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size
            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = functional.conv2d(gt_map.view(1, 1,self.fine_size, self.fine_size), self.kernel, bias=None, stride=2, padding=0)
            return img.view(3, self.fine_size, self.fine_size), gt_map.view(1, self.fine_size//2, self.fine_size//2)

    def __len__(self):
        return self.train_num


#
# For evalation, we also return img_path.
# This help get the paths of '.mat' recording the real num(not from density map).
class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 mode="crop",
                 dataset_name="QNRF",
                 device=None,
                 ):
        super(EvalDatasetConstructor, self).__init__()
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))

        self.validate_num = len(imgs)
        print('Constructing testing dataset...')
        for i in range(self.validate_num):
            img_tmp = imgs[i]
            den = os.path.join(self.gt_root, os.path.splitext(os.path.basename(img_tmp))[0] + ".npy")
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den])


    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path = self.imgs[index]

            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
            img = super(EvalDatasetConstructor, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))

            img_shape, gt_shape = img.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            patch_h, patch_w = imgs.size(2), imgs.size(3)
            gt_H, gt_W = gt_shape[1] // 2, gt_shape[2] // 2
            return img_path, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), torch.tensor(gt_H), torch.tensor(gt_W), torch.tensor(patch_h), torch.tensor(patch_w)

    def __len__(self):
        return self.validate_num
