import os
import glob
import numpy as np

from imageio import imread
from scipy.spatial.transform import Rotation
from lib.misc.pano_lsd_align import rotatePanorama

import torch
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, dmin=0.01, dmax=10, hw=(512, 1024),
            rand_rotate=False, rand_flip=False, rand_gamma=False,
            rand_pitch=0, rand_roll=0,
            fix_pitch=0, fix_roll=0):
        self.fname = []
        self.rgb_paths, self.d_paths = [], []
        self.dmin = dmin
        self.dmax = dmax
        self.hw = hw
        self.rand_rotate = rand_rotate
        self.rand_flip = rand_flip
        self.rand_gamma = rand_gamma
        self.rand_pitch = rand_pitch
        self.rand_roll = rand_roll
        self.fix_pitch = fix_pitch
        self.fix_roll = fix_roll

    def __len__(self):
        return len(self.rgb_paths)

    def read_rgb(self, path):
        return imread(path)

    def read_depth(self, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        # Read data
        fname = self.fname[idx]
        color = self.read_rgb(self.rgb_paths[idx])
        depth = self.read_depth(self.d_paths[idx])

        # To tensor and reshape to [C, H, W]
        color = torch.from_numpy(color).permute(2,0,1).float() / 255
        depth = torch.from_numpy(depth)[None].float()
        depth = torch.clamp(depth, max=self.dmax)

        # Resize
        if color.shape[1:] != self.hw:
            color = torch.nn.functional.interpolate(color[None], self.hw, mode='area')[0]
        if depth.shape[1:] != self.hw:
            depth = torch.nn.functional.interpolate(depth[None], self.hw, mode='nearest')[0]

        # Data augmentation
        if self.rand_rotate:
            shift = np.random.randint(self.hw[1])
            color = torch.roll(color, shift, dims=-1)
            depth = torch.roll(depth, shift, dims=-1)

        if self.rand_flip and np.random.randint(2):
            color = torch.flip(color, dims=[-1])
            depth = torch.flip(depth, dims=[-1])

        if self.rand_gamma:
            p = np.random.uniform(1, 1.2)
            if np.random.randint(2) == 0:
                p = 1 / p
            color = color ** p

        # Rotation augmentation
        if self.rand_pitch > 0 or self.rand_roll > 0 or self.fix_pitch != 0 or self.fix_roll > 0:
            color = color.permute(1,2,0).numpy()
            depth = depth.permute(1,2,0).numpy()
            if self.fix_pitch:
                rot = self.fix_pitch
                vp = Rotation.from_rotvec([rot * np.pi / 180, 0, 0]).as_matrix()
                color = rotatePanorama(color, vp, order=0)
            elif self.rand_pitch > 0:
                rot = np.random.randint(0, self.rand_pitch)
                vp = Rotation.from_rotvec([rot * np.pi / 180, 0, 0]).as_matrix()
                color = rotatePanorama(color, vp, order=0)
                depth = rotatePanorama(depth, vp, order=0)
            if self.fix_roll:
                rot = self.fix_roll
                vp = Rotation.from_rotvec([0, rot * np.pi / 180, 0]).as_matrix()
                color = rotatePanorama(color, vp, order=0)
            elif self.rand_roll > 0:
                rot = np.random.randint(0, self.rand_roll)
                vp = Rotation.from_rotvec([0, rot * np.pi / 180, 0]).as_matrix()
                color = rotatePanorama(color, vp, order=0)
                depth = rotatePanorama(depth, vp, order=0)
            color = torch.from_numpy(color).permute(2,0,1).float()
            depth = torch.from_numpy(depth).permute(2,0,1).float()

        return {'x': color, 'depth': depth, 'fname': fname.ljust(200)}


class CorruptMP3dDepthDataset(BaseDataset):
    def __init__(self, root, scene_txt, **kwargs):
        super(CorruptMP3dDepthDataset, self).__init__(**kwargs)

        # List all rgbd paths
        with open(scene_txt) as f:
            scene_split_ids = set(f.read().split())
        for scene in os.listdir(root):
            scene_root = os.path.join(root, scene)
            if not os.path.isdir(scene_root) or scene not in scene_split_ids:
                continue
            for cam in os.listdir(scene_root):
                cam_root = os.path.join(scene_root, cam)
                if not os.path.isdir(cam_root):
                    continue
                self.rgb_paths.append(os.path.join(cam_root, 'color.jpg'))
                self.d_paths.append(os.path.join(cam_root, 'depth.npy'))
        assert len(self.rgb_paths) == len(self.d_paths)
        for path in self.rgb_paths:
            self.fname.append('_'.join(path.split('/')))

    def read_depth(self, path):
        depth = np.load(path)
        depth[depth == 0.01] = 0
        return depth


class MP3dDepthDataset(BaseDataset):
    def __init__(self, root, scene_txt, **kwargs):
        super(MP3dDepthDataset, self).__init__(**kwargs)

        # List all rgbd paths
        with open(scene_txt) as f:
            scene_split_ids = set(f.read().split())
        for scene in os.listdir(root):
            scene_root = os.path.join(root, scene)
            if not os.path.isdir(scene_root) or scene not in scene_split_ids:
                continue
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(scene_root, '*rgb.png'))))
            self.d_paths.extend(sorted(glob.glob(os.path.join(scene_root, '*depth.exr'))))
        assert len(self.rgb_paths) == len(self.d_paths)
        for path in self.rgb_paths:
            self.fname.append('_'.join(path.split('/')))

    def read_depth(self, path):
        import Imath
        import OpenEXR
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        depth = np.frombuffer(f.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
        depth = depth.reshape(size[1], size[0])
        f.close()
        return depth.astype(np.float32)


class S2d3dDepthDataset(BaseDataset):
    def __init__(self, root, scene_txt, **kwargs):
        super(S2d3dDepthDataset, self).__init__(**kwargs)

        # List all rgbd paths
        with open(scene_txt) as f:
            path_pair = [l.strip().split() for l in f]
        for rgb_path, dep_path in path_pair:
            self.rgb_paths.append(os.path.join(root, rgb_path))
            self.d_paths.append(os.path.join(root, dep_path))
            self.fname.append(os.path.split(rgb_path)[1])

    def read_depth(self, path):
        depth = imread(path)
        return np.where(depth==65535, 0, depth/512)

