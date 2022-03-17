import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random, json
import os.path
from collections import OrderedDict
import sys
import numpy as np
import skimage.io as io
import pdb
# import dlib

import torch

from .FileTools import _image_file, _all_images, _video_image_file, _sample_from_videos_frames, sample_info_video, video_frame_names
from .Loaders import pil_loader, load_to_tensor, to_pil_image, to_tensor
from .Prepro import _id, random_pre_process, random_pre_process_pair

from ..Functions import functional as Func
# from ..TorchNet.RLSR import random_gaussian_kernel
from ..TorchNet.GaussianKernels import random_gaussian_kernel
from ..Functions.functional import random_affine, affine_im, edge_mask

def _mod_crop(im, scala):
    w, h = im.size
    return im.crop((0, 0, w - w % scala, h - h % scala))


class SRDataSet(data.Dataset):
    """
    DataSet for small images, easy to read
    do not need buffer
    random crop.
    all the image are same size
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 image_size,
                 scala=2,
                 interp=Image.BICUBIC,
                 mode='Y',
                 sub_dir=False,
                 prepro=random_pre_process):
        """
            :param data_path: Path to data root
            :param lr_patch_size: the Low resolution size, by default, the patch is square
            :param scala: SR scala, default is 4
            :param interp: interpolation for resize, default is Image.BICUBIC, optional [Image.BILINEAR, Image.BICUBIC]
            :param mode: 'RGB' or 'Y'
            :param sub_dir: if True, then all the images in the `data_path` directory AND child directory will be use
            :param prepro: function fo to ``PIL.Image``!, will run this function before crop and resize
        """
        data_path = os.path.abspath(data_path)
        print('Initializing DataSet, data root: %s ...' % data_path)
        if sub_dir:
            self.image_file_list = _all_images(data_path)
        else:
            self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        assert lr_patch_size * scala <= image_size, "Wrong size."
        self.lr_size = lr_patch_size
        self.image_size = image_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        if self.mode == 'Y':
            image = pil_loader(self.image_file_list[index], mode='YCbCr')
        else:
            image = pil_loader(self.image_file_list[index], mode=self.mode)
        hr_img = Func.random_crop(self.prepro(image), self.crop_size)
        lr_img = Func.resize(hr_img, self.lr_size, interpolation=self.interp)
        if self.mode == 'Y':
            return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
        else:
            return Func.to_tensor(lr_img), Func.to_tensor(hr_img)

    def __len__(self):
        return len(self.image_file_list)


class SRDataLarge(data.Dataset):
    """
    DataSet for Large images, hard to read once
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=2,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 sub_dir=False,
                 prepro=random_pre_process,
                 buffer=4):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        data_path = os.path.abspath(data_path)
        print('Initializing DataSet, data root: %s ...' % data_path)
        if sub_dir:
            self.image_file_list = _all_images(data_path)
        else:
            self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        self.buffer = buffer
        self.current_image_index = -1
        self.current_patch_index = -1
        self.current_image = None

    def __len__(self):
        return len(self.image_file_list) * self.buffer

    def __getitem__(self, index):
        image_index = index // self.buffer
        if self.current_image_index != image_index:
            if self.mode == 'Y':
                image = pil_loader(self.image_file_list[image_index], mode='YCbCr')
            else:
                image = pil_loader(self.image_file_list[image_index], mode=self.mode)
            self.current_image = self.prepro(image)
            self.current_image_index = image_index
        w, h = self.current_image.size
        cropable = (w - self.lr_size * self.scala, h - self.lr_size * self.scala)
        cw = random.randrange(0, cropable[0])
        ch = random.randrange(0, cropable[1])
        hr_img = Func.crop(self.current_image, cw, ch, self.lr_size * self.scala, self.lr_size * self.scala)
        lr_img = Func.resize(hr_img, self.lr_size, interpolation=self.interp)
        if self.mode == 'Y':
            return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
        else:
            return Func.to_tensor(lr_img), Func.to_tensor(hr_img)


class SRDataList(data.Dataset):
    """
    DataSet for Large images, hard to read once
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    load image from name.txt which contains all images` paths
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=4,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 transform=None,
                 prepro=random_pre_process,
                 train=True,
                 need_name=False,
                 rgb_range=1.,
                 ):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        # self.current_image = None
        self.train = train
        self.transform = transform
        self.need_name = need_name
        self.rgb_range = rgb_range

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        data = {}
        if self.need_name:
            im_path = self.image_file_list[index].strip('\n')
            data['HR_PATH'] = im_path

        if self.mode == 'Y':
            image = pil_loader(self.image_file_list[index].strip('\n'), mode='YCbCr')
        else:
            image = pil_loader(self.image_file_list[index], mode=self.mode)

        if self.train:
            image = self.prepro(image)

        if self.train:
            hr_img = self.transform(image)
            w, h = hr_img.size
            lr_img = Func.resize(hr_img, w // self.scala, interpolation=self.interp)
        else:
            w, h = image.size
            hr_img = _mod_crop(image, self.scala)
            lr_img = Func.resize(hr_img, (w // self.scala, h // self.scala), interpolation=self.interp)

        if self.mode == 'Y':
            data['LR'] = Func.to_tensor(lr_img)[:1] * self.rgb_range
            data['HR'] = Func.to_tensor(hr_img)[:1] * self.rgb_range
        else:
            data['LR'] = Func.to_tensor(lr_img) * self.rgb_range
            data['HR'] = Func.to_tensor(hr_img) * self.rgb_range

        return data


class SRMDDataset(data.Dataset):
    """
    DataSet for Large images, hard to read once
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    load image from name.txt which contains all images` paths
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=4,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 transform=None,
                 prepro=random_pre_process,
                 # buffer=4
                 train=True,
                 need_name=False
                 ):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        # if sub_dir:
        #     self.image_file_list = _all_images(data_path)
        # else:
        #     self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        self.transform = transform
        self.need_name = need_name

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):

        if self.need_name:
            file_name = self.image_file_list[index].strip('\n')
        if self.mode == 'Y':
            image = pil_loader(self.image_file_list[index].strip('\n'), mode='YCbCr')
        else:
            image = pil_loader(self.image_file_list[index], mode=self.mode)

        if self.train:
            self.current_image = self.prepro(image)
        else:
            self.current_image = image
        w, h = self.current_image.size

        hr_img = self.transform(self.current_image) if self.train else _mod_crop(self.current_image, self.scala)
        # else:
        #     hr_img = mod_crop(self.current_image, self.scala)

        if self.mode == 'Y':
            if self.need_name:
                return Func.to_tensor(hr_img)[:1], file_name
            else:
                return Func.to_tensor(hr_img)[:1]
        else:
            if self.need_name:
                return Func.to_tensor(hr_img), file_name
            else:
                return Func.to_tensor(hr_img)

        # if self.mode == 'Y':
        #     return Func.to_tensor(hr_img)[:1], file_name if self.need_name else Func.to_tensor(hr_img)[:1]
        # else:
        #     return Func.to_tensor(hr_img), file_name if self.need_name else Func.to_tensor(hr_img)


# TODO: SRDataLargeGrid(data.Dataset)
# class SRDataLargeGrid(data.Dataset):
#     """
#     DataSet for Large images, hard to read once
#     need buffer
#     need to crop, but crop by grid
#     all the image are Big size (DIV2K for example)
#     """


class SRDataListAC(data.Dataset):
    """
    DataSet for Large images, load all to memory
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    load image from name.txt which contains all images` paths
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=4,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 transform=None,
                 prepro=random_pre_process,
                 train=True,
                 need_lr=True
                 ):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.image_files = []

        for i in range(len(self.image_file_list)):
            im_path = self.image_file_list[i]
            im = pil_loader(im_path.strip('\n'), mode='YCbCr' if mode == 'Y' else mode)
            self.image_files.append(im)
            # if i % 100 == 0 or i == len(self.image_file_list) - 1:
            print('loading: [%d/%d] %s' % (i + 1, len(self.image_file_list), im_path), end='\r')
            sys.stdout.flush()

        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        self.transform = transform
        self.need_lr = need_lr

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image = self.image_files[index]
        lr_img = None
        if self.train:
            hr_img = self.transform(image)
            hr_img = self.prepro(hr_img)
            w, h = hr_img.size
            if self.need_lr:
                lr_img = Func.resize(hr_img, w // self.scala, interpolation=self.interp)
        else:
            hr_img = _mod_crop(image, self.scala)
            w, h = hr_img.size
            if self.need_lr:
                lr_img = Func.resize(hr_img, (w // self.scala, h // self.scala), interpolation=self.interp)
        if self.mode == 'Y':
            if self.need_lr:
                return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
            else:
                return Func.to_tensor(hr_img)[:1]
        else:
            if self.need_lr:
                return Func.to_tensor(lr_img), Func.to_tensor(hr_img)
            else:
                return Func.to_tensor(hr_img)


class SRDataDoubleScaleAC(data.Dataset):
    """
    DataSet for Large images, load all to memory
    need to random crop
    all the image are Big size (DIV2K for example)
    load image from Image path
    double scale ops for one image, e.g. downsample x(scala * down_factor), upsample x down_factor
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=2,
                 down_factor=2,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 transform=None,
                 prepro=random_pre_process,
                 train=True
                 ):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala (down, up)
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.image_files = []

        for i in range(len(self.image_file_list)):
            im_path = self.image_file_list[i]
            im = pil_loader(im_path.strip('\n'), mode='YCbCr' if mode == 'Y' else mode)
            self.image_files.append(im)
            print('loading: [%d/%d] %s' % (i + 1, len(self.image_file_list), im_path))
            sys.stdout.flush()

        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        self.transform = transform
        self.down_factor = down_factor

    def __len__(self):
        return len(self.image_file_list)

    def _mod_crop(self, im, scala):
        w, h = im.size
        if w % scala == 0 and h % scala == 0:
            return im
        else:
            return im.crop(0, 0, w - w % scala, h - h % scala)

    def __getitem__(self, index):
        image = self.image_files[index]
        if self.train:
            hr_img = self.transform(image)
            hr_img = self.prepro(hr_img)
            w, h = hr_img.size
            lr_img = Func.resize(hr_img, w // self.scala, interpolation=self.interp)
        else:
            hr_img = self._mod_crop(image, self.scala)
            w, h = hr_img.size
            lr_img = Func.resize(hr_img, (w // (self.scala * self.down_factor), h // (self.scala * self.down_factor)), interpolation=self.interp)
            lr_img = Func.resize(lr_img, (w // self.scala, h // self.scala), interpolation=self.interp)
        if self.mode == 'Y':
            return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
        else:
            return Func.to_tensor(lr_img), Func.to_tensor(hr_img)


class RealPairDataset(data.Dataset):
    """
    Dataset for Real HL Pair
    Add Param For Image Enhance: No Change in Resolution
    :param rgb_range: 255. for RCAN and EDSR, 1. for others
    :param need_hr_down: For Degrade Train, From HR_down to LR;
    :param need_lr_up: For Upgrade Train, From LR_up to HR;
    :param need_edge_mask: a mask based on edge detect algorithm, to augment edge loss;
    :param multiHR: for HGSR, which need multi HR for intermediate supervision;
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 need_hr_down=False,
                 need_lr_up=False,
                 need_edge_mask=False,
                 need_name=False,
                 rgb_range=1.,
                 multiHR=False,
                 edge_mask_ch=1,
                 need_sr=False
                 ):
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        if need_sr:
            sr_file_path = os.path.join(pair_folder_path, 'train_SR') if train else os.path.join(pair_folder_path, 'test_SR')
            self.sr_file_list = _all_images(sr_file_path)
            print('Initializing DataSet, image list: %s ...' % pair_folder_path)
            print('Found %d HR %d LR %d SR...' % (len(self.hr_file_list), len(self.lr_file_list), len(self.sr_file_list)))
        else:
            print('Initializing DataSet, image list: %s ...' % pair_folder_path)
            print('Found %d HR %d LR ...' % (len(self.hr_file_list), len(self.lr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        self.need_name = need_name
        self.scale = scala
        self.need_hr_down = need_hr_down
        self.need_lr_up = need_lr_up
        self.need_edge_mask = need_edge_mask
        self.rgb_range = rgb_range
        self.multiHR = multiHR
        self.edge_mask_ch = edge_mask_ch
        self.need_sr = need_sr

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        data = {}
        # Return Image Path
        if self.need_name:
            data['HR_PATH'] = self.hr_file_list[index].strip('\n')
            data['LR_PATH'] = self.lr_file_list[index].strip('\n')
            if self.need_sr:
                data['SR_PATH'] = self.sr_file_list[index].strip('\n')

        # For Color Mode
        if self.mode == 'Y':
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')
            if self.need_sr:
                sr_img = pil_loader(self.sr_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')
            if self.need_sr:
                sr_img = pil_loader(self.sr_file_list[index].strip('\n'), mode='RGB')

        # For Train or Test, Whether Crop/Rotate Image
        if self.train:
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)
        else:
            hr_patch, lr_patch = hr_img, lr_img
            if self.need_sr:
                sr_patch = sr_img

        # For Multi-Scale Training, Need HR downsample
        if self.need_hr_down:
            w, h = lr_patch.size
            hr_patch_down = hr_patch.resize((w, h), resample=Image.BICUBIC)
        if self.need_lr_up:
            w, h = hr_patch.size
            lr_patch_up = lr_patch.resize((w, h), resample=Image.BICUBIC)

        # For HGSR, Need Multi Downsampled HR
        if self.multiHR:
            w, h = lr_patch.size
            hr_down_1 = hr_patch.resize((w, h), resample=Image.BICUBIC)
            if self.scale == 4:
                hr_down_2 = hr_patch.resize((w * 2, h * 2), resample=Image.BICUBIC)
            else:
                hr_down_2 = hr_patch

        # Image To Tensor
        if self.mode == 'Y':
            data['LR'] = Func.to_tensor(lr_patch)[:1] * self.rgb_range
            data['HR'] = Func.to_tensor(hr_patch)[:1] * self.rgb_range
            if self.need_sr:
                data['SR'] = Func.to_tensor(sr_patch)[:1] * self.rgb_range
            if self.need_hr_down:
                data['HR_DOWN'] = Func.to_tensor(hr_patch_down)[:1]
            if self.need_lr_up:
                data['LR_UP'] = Func.to_tensor(lr_patch_up)[:1]
            if self.multiHR:
                data['HR_D1'] = Func.to_tensor(hr_down_1)[:1]
                data['HR_D2'] = Func.to_tensor(hr_down_2)[:1]

        else:
            data['LR'] = Func.to_tensor(lr_patch) * self.rgb_range
            data['HR'] = Func.to_tensor(hr_patch) * self.rgb_range
            if self.need_sr:
                data['SR'] = Func.to_tensor(sr_patch) * self.rgb_range
            if self.need_hr_down:
                data['HR_DOWN'] = Func.to_tensor(hr_patch_down)
            if self.need_lr_up:
                data['LR_UP'] = Func.to_tensor(lr_patch_up)
            if self.multiHR:
                data['HR_D1'] = Func.to_tensor(hr_down_1)
                data['HR_D2'] = Func.to_tensor(hr_down_2)

        if self.need_edge_mask:
            data['HR_EDGE_MAP'] = edge_mask(hr_patch, ch=self.edge_mask_ch)
            data['LR_EDGE_MAP'] = edge_mask(lr_patch, ch=self.edge_mask_ch)

        return data


class RealPairDatasetAC(data.Dataset):
    """
    Dataset for real pair HR/LR, which has train_HR/train_LR folders in data_path
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 ):
        self.train = train
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        self.lr_files = []
        self.hr_files = []

        for i in range(len(self.lr_file_list)):
            lr_file = self.lr_file_list[i]
            hr_file = self.hr_file_list[i]
            lr_img = Image.open(lr_file)
            hr_img = Image.open(hr_file)
            self.lr_files.append(lr_img)
            self.hr_files.append(hr_img)
            if i % 100 == 0 or i == len(self.lr_file_list) - 1:
                print('loading: [%d/%d] %s' % (i + 1, len(self.lr_file_list), hr_file))
                sys.stdout.flush()


        self.hr_file_list = _all_images(hr_file_path)
        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.scale = scala

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        hr_img = self.hr_files[index]
        lr_img = self.lr_files[index]

        if self.train:
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        if self.mode == 'Y':
            hr_patch = hr_patch.convert('YCbCr')
            lr_patch = lr_patch.convert('YCbCr')
            return Func.to_tensor(lr_patch)[:1], Func.to_tensor(hr_patch)[:1]
        else:
            return Func.to_tensor(lr_patch), Func.to_tensor(hr_patch)


class MixRealBicDataset(data.Dataset):
    """
    Dataset for HR and mix real or bicubic LR, which has train_HR/train_LR folders in data_path
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 real_rate=0.9,
                 ):
        self.real_rate = real_rate
        self.train = train
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        self.lr_files = []
        self.hr_files = []

        # for i in range(len(self.lr_file_list)):
        #     lr_file = self.lr_file_list[i]
        #     hr_file = self.hr_file_list[i]
        #     lr_img = Image.open(lr_file)
        #     hr_img = Image.open(hr_file)
        #     self.lr_files.append(lr_img)
        #     self.hr_files.append(hr_img)
        #     if i % 100 == 0 or i == len(self.lr_file_list) - 1:
        #         print('loading: [%d/%d] %s' % (i + 1, len(self.lr_file_list), hr_file))
        #         sys.stdout.flush()

        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.scale = scala

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        if self.mode == 'Y':
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')

        if self.train:
            flag = torch.ones(1)
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)

            r = random.random()
            if r > self.real_rate:

                # print('%f, Bic' % r)
                flag *= 0

                w, h = hr_patch.size
                lr_patch = hr_patch.resize((w // self.scale, h // self.scale), resample=Image.BICUBIC)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        if self.mode == 'Y':
            hr_patch = hr_patch.convert('YCbCr')
            lr_patch = lr_patch.convert('YCbCr')
            return Func.to_tensor(lr_patch)[:1], Func.to_tensor(hr_patch)[:1]
        else:
            return Func.to_tensor(lr_patch), Func.to_tensor(hr_patch), flag


class MixRealBicDatasetAC(data.Dataset):
    """
    Dataset for HR and mix real or bicubic LR, which has train_HR/train_LR folders in data_path
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 real_rate=0.95,
                 ):
        self.real_rate = real_rate
        self.train = train
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        self.lr_files = []
        self.hr_files = []

        for i in range(len(self.lr_file_list)):
            lr_file = self.lr_file_list[i]
            hr_file = self.hr_file_list[i]
            lr_img = Image.open(lr_file)
            hr_img = Image.open(hr_file)
            self.lr_files.append(lr_img)
            self.hr_files.append(hr_img)
            if i % 100 == 0 or i == len(self.lr_file_list) - 1:
                print('loading: [%d/%d] %s' % (i + 1, len(self.lr_file_list), hr_file), end='\r')
                sys.stdout.flush()

        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.scale = scala

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        hr_img = self.hr_files[index]
        lr_img = self.lr_files[index]

        if self.train:
            flag = torch.ones(1)
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)

            r = random.random()
            if r > self.real_rate:
                flag *= 0

                w, h = hr_patch.size
                lr_patch = hr_patch.resize((w // self.scale, h // self.scale), resample=Image.BICUBIC)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        if self.mode == 'Y':
            hr_patch = hr_patch.convert('YCbCr')
            lr_patch = lr_patch.convert('YCbCr')
            return Func.to_tensor(lr_patch)[:1], Func.to_tensor(hr_patch)[:1]
        else:
            return Func.to_tensor(lr_patch), Func.to_tensor(hr_patch), flag


class RandDeformDatasetAC(data.Dataset):
    """
    DataSet for STN, return HR and HR_trans, which been randomly spin and shifted n pixels
    """
    def __init__(self,
                 data_path,
                 patch_size,
                 # scala=4,
                 # interp=Image.BICUBIC,
                 mode='RGB',
                 # transform=None,
                 # prepro=random_pre_process,
                 train=True,
                 # need_lr=True
                 degree=8,
                 translate=5,
                 trans_scale=0.08,
                 ):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.image_files = []

        for i in range(len(self.image_file_list)):
            im_path = self.image_file_list[i]
            im = pil_loader(im_path.strip('\n'), mode='YCbCr' if mode == 'Y' else mode)
            self.image_files.append(im)
            if i % 50 == 0 or i == len(self.image_file_list) - 1:
                print('loading: [%d/%d] %s' % (i + 1, len(self.image_file_list), im_path))
                sys.stdout.flush()

        self.patch_size = patch_size
        self.mode = mode
        self.train = train
        self.random_crop = transforms.RandomCrop(patch_size)
        #
        self.degree = degree
        self.translate = translate
        self.trans_scale = trans_scale

        self.train = train

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image = self.image_files[index]

        if self.train:
            patch = self.random_crop(image)
            deformed_patch = random_affine(patch, degrees=self.degree, translate=self.translate, s_scale=self.trans_scale)
        else:
            patch = image
            deformed_patch = affine_im(patch, degrees=self.degree, translate=self.translate, s_scale=self.trans_scale)

        return Func.to_tensor(patch), Func.to_tensor(deformed_patch)


class C2FDataset(data.Dataset):
    """
    Dataset for Coarse to Fine(Multi-Scale SR), need LR Downscale, LR and HR
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 ):
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        # self.need_name = need_name
        self.scale = scala

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        # file_name = os.path.basename(self.lr_file_list[index].strip('\n'))
        # file_name = file_name.split('.')[0]

        if self.mode == 'Y':
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')

        if self.train:
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        w, h = lr_patch.size
        lr_down_patch = lr_patch.resize((w // self.scale, h // self.scale), resample=Image.BICUBIC)

        if self.mode == 'Y':
            return to_tensor(lr_down_patch)[:1], to_tensor(lr_patch)[:1], to_tensor(hr_patch)[:1]
        else:
            return to_tensor(lr_down_patch), to_tensor(lr_patch), to_tensor(hr_patch)


class RealPairDataset_mod1(data.Dataset):
    """
    Dataset for Real HL Pair
    Use Dict to Trans Variable
    Add Param For Image Enhance: No Change in Resolution
    :param need_hr_down: For Degrade Train, From HR_down to LR
    :param need_lr_up: For Upgrade Train, From LR_up to HR,
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 need_hr_down=False,
                 need_lr_up=False,
                 ):
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        # self.need_name = need_name
        self.scale = scala
        self.need_hr_down = need_hr_down
        self.need_lr_up = need_lr_up

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        # file_name = os.path.basename(self.lr_file_list[index].strip('\n'))
        # file_name = file_name.split('.')[0]
        if self.mode == 'Y':
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')

        if self.train:
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        if self.need_hr_down:
            w, h = lr_patch.size
            hr_patch_down = hr_patch.resize((w, h), resample=Image.BICUBIC)

        if self.need_lr_up:
            w, h = hr_patch.size
            lr_patch_up = lr_patch.resize((w, h), resample=Image.BICUBIC)

        if self.mode == 'Y':
            data = [Func.to_tensor(lr_patch)[:1], Func.to_tensor(hr_patch)[:1]]
            if self.need_hr_down:
                data.append(Func.to_tensor(hr_patch_down)[:1])
            if self.need_lr_up:
                data.append(Func.to_tensor(lr_patch_up)[:1])
        else:
            data = [Func.to_tensor(lr_patch), Func.to_tensor(hr_patch)]
            if self.need_hr_down:
                data.append(Func.to_tensor(hr_patch_down))
            if self.need_lr_up:
                data.append(Func.to_tensor(lr_patch_up))
        return data


class SRDataset(data.Dataset):
    """
    Dataset for SR, need LR/HR,
    LR Image is created by downsampling HR Image with Interpolation
    """
    def __init__(self,
                 data_path,
                 lr_psize,
                 scala=4,
                 interp=Image.BICUBIC,
                 cmode='RGB',
                 transform=None,
                 prepro=random_pre_process,
                 train=True,
                 im_path=False,
                 need_lr=True,
                 ):
        """

        :param data_path:
        :param lr_psize:
        :param scala:
        :param interp:
        :param mode:
        :param transform:
        :param prepro:
        :param train:
        :param need_name:
        :param need_lr:
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_psize
        self.scala = scala
        self.interp = interp
        self.gray = True if cmode == 'Y' else False
        self.hr_psize = lr_psize * scala
        self.prepro = prepro
        self.train = train
        self.transform = transform
        self.need_lr = need_lr

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        data = {}
        im_path = self.image_file_list[index].strip('\n')
        data['path'] = im_path
        if self.gray:
            hr_img = pil_loader(self.image_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.image_file_list[index], mode='RGB')

        if self.train:
            hr_img = self.prepro(self.transform(hr_img))

        hr_tensor = Func.to_tensor(hr_img)
        hr_tensor = hr_tensor[:1] if self.gray else hr_tensor
        data['hr'] = hr_tensor

        if self.need_lr:
            hr_img = _mod_crop(hr_img, self.scala)
            w, h = hr_img
            lr_img = Func.resize(hr_img, (w // self.scala, h // self.scala), interpolation=self.interp)
            lr_tensor = Func.to_tensor(lr_img)
            lr_tensor = lr_tensor[:1] if self.gray else lr_tensor
            data['lr'] = lr_tensor

        return data


class RealPairDatasetOri(data.Dataset):
    """
    Dataset for Real HL Pair
    Add Param For Image Enhance: No Change in Resolution
    :param need_hr_down: For Degrade Train, From HR_down to LR
    :param need_lr_up: For Upgrade Train, From LR_up to HR,
    """
    def __init__(self,
                 pair_folder_path,
                 lr_patch_size,
                 mode='RGB',
                 scala=4,
                 prepro=random_pre_process_pair,
                 train=True,
                 need_hr_down=False,
                 need_lr_up=False,
                 need_edge_mask=False,
                 need_name=False,
                 rgb_range=1.,
                 ):
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)
        self.hr_file_list = _all_images(hr_file_path)
        print('Initializing DataSet, image list: %s ...' % pair_folder_path)
        print('Found %d HR %d LR ...' % (len(self.lr_file_list), len(self.hr_file_list)))
        self.lr_size = lr_patch_size
        self.mode = mode
        self.hr_size = lr_patch_size * scala
        self.prepro = prepro
        self.current_image = None
        self.train = train
        self.need_name = need_name
        self.scale = scala
        self.need_hr_down = need_hr_down
        self.need_lr_up = need_lr_up
        self.need_edge_mask = need_edge_mask
        self.rgb_range = rgb_range

    def __len__(self):
        return len(self.lr_file_list)

    def __getitem__(self, index):
        # Return Image Path
        # For Color Mode
        file_name = os.path.basename(self.hr_file_list[index].strip('\n'))
        if self.mode == 'Y':
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')
        else:
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')

        # For Train or Test, Whether Crop/Rotate Image
        if self.train:
            hr_patch, lr_patch = self.prepro(hr_img, lr_img, self.lr_size, self.scale)
        else:
            hr_patch, lr_patch = hr_img, lr_img

        # For Multi-Scale Training, Need HR downsample
        if self.need_hr_down:
            w, h = lr_patch.size
            hr_patch_down = hr_patch.resize((w, h), resample=Image.BICUBIC)
        if self.need_lr_up:
            w, h = hr_patch.size
            lr_patch_up = lr_patch.resize((w, h), resample=Image.BICUBIC)

        # Image To Tensor
        if self.mode == 'Y':
            lr_patch = Func.to_tensor(lr_patch)[:1] * self.rgb_range
            hr_patch = Func.to_tensor(hr_patch)[:1] * self.rgb_range
            return lr_patch, hr_patch, file_name
            # data = [Func.to_tensor(lr_patch)[:1], Func.to_tensor(hr_patch)[:1]]
        else:
            lr_patch = Func.to_tensor(lr_patch) * self.rgb_range
            hr_patch = Func.to_tensor(hr_patch) * self.rgb_range
            return lr_patch, hr_patch, file_name


"""================= Previous Datasets ====================="""

class PreLoad(data.Dataset):
    def __init__(self, data_path, gt_path):
        self.data = torch.load(data_path)
        self.gt = torch.load(gt_path)
        assert len(self.data) == len(self.gt), "Wrong"
        self.leng = len(self.data)
        print('Data loaded, %d' % self.leng)

    def __getitem__(self, item):
        return self.data[item], self.gt[item]

    def __len__(self):
        return self.leng


class SimpleImageData(data.Dataset):
    def __init__(self, data_root, image_size=32, scale=1, loader=pil_loader, mode='Y', crop=True, prepro=random_pre_process):
        print('Initializing DataSet, data root: %s ...' % data_root)
        self.data = data_root
        self.img_list = _all_images(self.data)
        print('Found %d Images...' % len(self.img_list))
        self.size = image_size
        self.loader = loader
        self.mode = mode
        self.scale = scale
        if hasattr(prepro, '__call__'):
            self.prepro = prepro
        else:
            self.prepro = _id
        self.crop_size = crop

    def _return_mode(self, pil):
        if self.mode == 'Y':
            return Func.to_tensor(pil)[:1, :, :]
        else:
            return Func.to_tensor(pil)

    def __getitem__(self, index):
        pil = self.prepro(self.loader(self.img_list[index]))
        if self.scale != 1:
            pil = Func.resize(pil, (pil.size[0] // self.scale, pil.size[1] // self.scale), interpolation=Image.BICUBIC)
        if self.crop_size:
            return self._return_mode(Func.random_crop(pil, self.size))
        else:
            return self._return_mode(pil)

    def __len__(self):
        return len(self.img_list)


class VideoData(data.Dataset):
    """
    Data Store Format:

    Data Folder
        |
        |-Video Folder
        |   |-Video Frames (images)
        |
        |-Video Folder
            |-Video Frames (images)

        ...

        |-Video Folder
            |-Video Frames (images)
    """
    def __init__(self, data_folder_root, scala, image_size_w, image_size_h, time_window=5, time_stride=2, loader=pil_loader, interp=Image.BICUBIC):
        super(VideoData, self). __init__()
        self.time_window = time_window
        self.time_stride = time_stride
        self.loader = loader

        video_list = os.listdir(data_folder_root)
        # number of videos
        self.n_videos = len(video_list)
        # number of frames for every video, list
        self.n_video_frames = [0] * self.n_videos
        # samples number for every video, list
        # sample number times videos number is the sample of the data set
        self.n_samples = [0] * self.n_videos
        # start sample of the video
        self.area_summed = [0] * self.n_videos
        # the path to every video folder
        self.video_folders = [None] * self.n_videos
        # 2-D list, path for every frame for every video of the data set
        self.frame_files = [None] * self.n_videos
        # Initial above by for loop
        for i in range(self.n_videos):
            video_folder = os.path.join(data_folder_root, video_list[i])
            self.video_folders[i] = video_folder
            frame_file_list = os.listdir(video_folder)
            frame_file_list.sort()
            self.n_video_frames[i] = len(frame_file_list)
            self.frame_files[i] = [None] * len(frame_file_list)
            for j in range(self.n_video_frames[i]):
                self.frame_files[i][j] = os.path.join(video_folder, frame_file_list[j])
            self.n_samples[i] = (self.n_video_frames[i] - self.time_window) // self.time_stride
            if i != 0:
                self.area_summed[i] = sum(self.n_samples[:i])

        self.toTensor = transforms.ToTensor()
        self.scala = transforms.Resize((image_size_h // scala, image_size_w // scala), interpolation=interp)
        self.crop = transforms.CenterCrop((image_size_h, image_size_w))

        self.buffer = [None] * self.time_window
        self.return_buffer = [None] * (self.time_window + 1)
        self.current_sample_index = None
        self.current_video_index = None

    def _index_parser(self, index):
        global_sample_index = index
        for i in range(self.n_videos):
            if self.area_summed[i] > global_sample_index:
                video_index = i - 1
                frame_index = global_sample_index - self.area_summed[video_index]
                return video_index, frame_index
        video_index = self.n_videos - 1
        frame_index = global_sample_index - self.area_summed[video_index]
        return video_index, frame_index

    def _load_reuse(self, video_index, sample_index):
        self.buffer[: self.time_window - self.time_stride] = self.buffer[self.time_stride:]
        frame_file_index_start = sample_index * self.time_stride
        frame_files = self.frame_files[video_index][frame_file_index_start + self.time_window - self.time_stride: frame_file_index_start + self.time_window]
        for i, file_path in enumerate(frame_files):
            self.buffer[self.time_window - self.time_stride + i] = self.loader(file_path)

    def _load_new(self, video_index, sample_index):
        frame_file_index_start = sample_index * self.time_stride
        frame_files = self.frame_files[video_index][frame_file_index_start: frame_file_index_start + self.time_window]
        for i, file_path in enumerate(frame_files):
            self.buffer[i] = self.loader(file_path)

    def _load_frames(self, video_index, sample_index):
        if (self.current_video_index == video_index) and (self.current_sample_index == (sample_index - 1)):
            self._load_reuse(video_index, sample_index)
        elif (self.current_video_index == video_index) and (self.current_sample_index == sample_index):
            pass
        else:
            self._load_new(video_index, sample_index)
        self.current_video_index = video_index
        self.current_sample_index = sample_index

    def __getitem__(self, index):
        video_index, sample_index = self._index_parser(index)
        self._load_frames(video_index, sample_index)
        for i in range(self.time_window):
            if i != (self.time_window // 2):
                self.return_buffer[i] = self.toTensor(self.scala(self.crop(self.buffer[i])))[:1]
            else:
                HR_center = self.crop(self.buffer[i])
                self.return_buffer[i] = self.toTensor(self.scala(HR_center))[:1]
                self.return_buffer[self.time_window] = self.toTensor(HR_center)[:1]
        return tuple(self.return_buffer)

    def __len__(self):
        return sum(self.n_samples)


class OpticalFlowData(data.Dataset):
    """
    This Dataset is for training optical flow
    """
    def __init__(self, path, stride=2, mode='YCbCr'):
        """
        :param path:
        :param stride: 1 or 2
        """
        self.stride = stride
        self.mode = mode
        self.video_frame_list = _video_image_file(path)
        self.num_videos = len(self.video_frame_list)
        self.num_frames = [0] * self.num_videos
        for i, video in enumerate(self.video_frame_list):
            self.num_frames[i] = len(video)
        self.num_samples = [0] * self.num_videos
        for i, frames in enumerate(self.num_frames):
            self.num_samples[i] = frames // stride
        self.area_summed = [0] * self.num_videos
        for i, frames in enumerate(self.num_samples):
            if i != 0:
                self.area_summed[i] = sum(self.num_samples[:i])

    def _index_parser(self, index):
        global_sample_index = index
        for i in range(self.num_videos):
            if self.area_summed[i] > global_sample_index:
                video_index = i - 1
                frame_index = global_sample_index - self.area_summed[video_index]
                return video_index, frame_index
        video_index = self.num_videos - 1
        frame_index = global_sample_index - self.area_summed[video_index]
        return video_index, frame_index

    def _load_frames(self, video_index, sample_index):
        frame_t = pil_loader(self.video_frame_list[video_index][sample_index * self.stride], mode=self.mode)
        frame_tp1 = pil_loader(self.video_frame_list[video_index][sample_index * self.stride + 1], mode=self.mode)
        return Func.to_tensor(frame_t)[:1], Func.to_tensor(frame_tp1)[:1]

    def __getitem__(self, index):
        video_index, frame_index = self._index_parser(index)
        return self._load_frames(video_index, frame_index)

    def __len__(self):
        return sum(self.num_samples)


class VideoFaceSRData(data.Dataset):

    def __init__(self, data_folder_root, gt_folder_root, time_window=5, time_stride=8, loader=pil_loader, mode='YCbCr'):
        self.time_window = time_window
        self.time_stride = time_stride
        self.loader = loader
        self.mode = mode
        self.data_root = data_folder_root
        self.gt_root = gt_folder_root
        self.video_frames = _video_image_file(os.path.abspath(data_folder_root))
        self.n_videos = len(self.video_frames)
        self.samples, self.area_sum_samples = sample_info_video(self.video_frames, self.time_window, self.time_stride)

    def _index_parser(self, index):
        global_sample_index = index
        for i in range(self.n_videos):
            if self.area_sum_samples[i] > global_sample_index:
                video_index = i - 1
                frame_index = global_sample_index - self.area_sum_samples[video_index]
                return video_index, frame_index
        video_index = self.n_videos - 1
        frame_index = global_sample_index - self.area_sum_samples[video_index]
        return video_index, frame_index

    def _load_frames(self, video_index, sample_index):
        frame_file_index_start = sample_index * self.time_stride
        frame_files = self.video_frames[video_index][frame_file_index_start: frame_file_index_start + self.time_window]
        hr_video, hr_frame = video_frame_names(frame_files[self.time_window // 2])
        hr_frame = os.path.join(os.path.join(self.gt_root, hr_video), hr_frame)
        return frame_files, hr_frame

    def __getitem__(self, index):
        video_index, sample_index = self._index_parser(index)
        load_list, hr_frame = self._load_frames(video_index, sample_index)
        buffer = [None] * self.time_window
        for i, frame in enumerate(load_list):
            buffer[i] = Func.to_tensor(self.loader(frame, mode=self.mode))[:1]
        hr = Func.to_tensor(self.loader(hr_frame, mode=self.mode))[:1]
        return buffer, hr

    def __len__(self):
        return sum(self.samples)


class SimpleCropVideoFaceSRData(VideoFaceSRData):

    def __init__(self, data_folder_root, gt_folder_root, dets_dict_root, LR_size=16, scala=8, time_window=5, time_stride=7, loader=pil_loader, mode='YCbCr'):
        super(SimpleCropVideoFaceSRData, self).__init__(data_folder_root, gt_folder_root, time_window=time_window, time_stride=time_stride, loader=loader, mode=mode)
        with open(dets_dict_root, 'r') as f:
            self.det16 = json.load(f)
        self.lr_size = LR_size
        self.scala = scala

    def __getitem__(self, index):
        video_index, sample_index = self._index_parser(index)
        load_list, hr_frame = self._load_frames(video_index, sample_index)
        buffer = [None] * self.time_window
        video_name, frame_name = video_frame_names(hr_frame)
        lr_bound, hr_bound = Func.crop_bound_correspong_L2H(self.det16[video_name][frame_name][5], lr_size=self.lr_size, up_scala=self.scala)
        for i, frame in enumerate(load_list):
            buffer[i] = Func.to_tensor(self.loader(frame, mode=self.mode).crop(lr_bound))[:1]
        hr = Func.to_tensor(self.loader(hr_frame, mode=self.mode).crop(hr_bound))[:1]
        return buffer, hr


class LoadSimpleCropVideoFaceSRData(VideoFaceSRData):

    def __init__(self, data_folder_root, gt_folder_root, dets_dict_root, LR_size=16, scala=8, time_window=5, time_stride=7, loader=pil_loader, mode='YCbCr'):
        super(LoadSimpleCropVideoFaceSRData, self).__init__(data_folder_root, gt_folder_root, time_window=time_window, time_stride=time_stride, loader=loader, mode=mode)
        with open(dets_dict_root, 'r') as f:
            self.det16 = json.load(f)
        self.lr_size = LR_size
        self.scala = scala
        self.buffer_dict = OrderedDict()

    def __getitem__(self, index):
        if index in self.buffer_dict.keys():
            return self.buffer_dict[index]
        else:
            video_index, sample_index = self._index_parser(index)
            load_list, hr_frame = self._load_frames(video_index, sample_index)
            buffer = [None] * self.time_window
            video_name, frame_name = video_frame_names(hr_frame)
            lr_bound, hr_bound = Func.crop_bound_correspong_L2H(self.det16[video_name][frame_name][5], lr_size=self.lr_size, up_scala=self.scala)
            for i, frame in enumerate(load_list):
                buffer[i] = Func.to_tensor(self.loader(frame, mode=self.mode).crop(lr_bound))[:1]
            hr = Func.to_tensor(self.loader(hr_frame, mode=self.mode).crop(hr_bound))[:1]
            return buffer, hr

















