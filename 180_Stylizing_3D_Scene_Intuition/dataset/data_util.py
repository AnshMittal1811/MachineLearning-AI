import os
import cv2
import glob
import torch
import math
import imageio
import random
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(256/480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
])

def find_files(dir, exts):
    # types should be ['*.png', '*.jpg']
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
    if len(files_grabbed) > 0:
        files_grabbed = sorted(files_grabbed)
    return files_grabbed
    
def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       img_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None,
                       style_imgs = None
                       ):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w
        self.img_path = img_path

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)
        self.style_imgs = style_imgs


    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = int(self.W_orig // resolution_level)
            self.H = int(self.H_orig // resolution_level)
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            
            # only load image at this time
            self.img = imageio.imread(self.img_path)[..., :3].astype(np.float32) / 255.0
            self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self.img = self.img.reshape((-1, 3))

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)

    def get_img(self):
        return self.img.reshape((self.H, self.W, 3))

    def get_style_img(self):
        return imageio.imread(self.style_img_path)[..., :3].astype(np.float32) / 255.0
        
    def get_style_input(self, mode=None, test_seed=None, style_ID=None):
        if mode == "test":
            # Fixed seed to choose the same style image for each GPU
            random.seed(test_seed)
            if style_ID != None:
                self.style_img_path = self.style_imgs[style_ID]
            else:
                self.style_img_path = random.sample(self.style_imgs, 1)[0]
        else:
            self.style_img_path = np.random.choice(self.style_imgs, 1)[0]
        
        ori_style_img = Image.open(self.style_img_path).convert('RGB')
        style_img = data_transform(ori_style_img)
        style_idx = torch.from_numpy(np.array([self.style_imgs.index(self.style_img_path)]))
        
        return style_img, style_idx

    def get_all(self):
        min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])
        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('rgb', self.img),
            ('min_depth', min_depth),
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, stage, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            if stage == "first":
                # Random from one image
                select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
            elif stage == "second":
                # Patch-wise training - Random choose one fixed pixel per region (region is random size and random position)
                total_inds = np.arange(self.H*self.W).reshape(self.H, self.W)
                patch_H, patch_W = 67, 81
                num_region_H, num_region_W = self.H//patch_H, self.W//patch_W #16, 24(Family, Francis, Horse), #8, 12(Truck, PG)
                
                region_size_v = np.random.randint(num_region_H//2, num_region_H+1)
                region_size_u = np.random.randint(num_region_W//3, num_region_W+1)
                region_position_v = np.random.randint(self.H - patch_H * region_size_v + region_size_v)
                region_position_u = np.random.randint(self.W - patch_W * region_size_u + region_size_u)
                select_inds = total_inds[region_position_v::region_size_v][:patch_H][:, region_position_u::region_size_u][:, :patch_W].reshape(-1)
            
        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]
        rgb = self.img[select_inds, :]          # [N_rand, 3]
        min_depth = 1e-4 * np.ones_like(rays_d[..., 0])
      
        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('rgb', rgb),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret
