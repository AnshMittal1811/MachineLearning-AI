import os
import torch
import numpy as np
import pickle
import os.path as osp

from data.robotcar_sdk.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from data.robotcar_sdk.camera_model import CameraModel
from data.robotcar_sdk.image import load_image as robotcar_loader
from tools.utils import process_poses, calc_vos_simple, load_image
from torch.utils import data
from functools import partial

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, mode=0, seed=7, real=False, skip_images=False, vo_lib='orbslam'):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        data_dir = osp.join(data_path, '7Scenes', scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(data_dir, 'train_split.txt')
        else:
            split_file = osp.join(data_dir, 'test_split.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                format(i))).flatten()[:12] for i in frame_idx]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 1:
                img = None
                while img is None:
                    img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 2:
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_image(self.c_imgs[index])
                    d_img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        return img, pose

    def __len__(self):
        return self.poses.shape[0]

class RobotCar(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, real=False, skip_images=False, seed=7, undistort=False, vo_lib='stereo'):
        np.random.seed(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undistort

        # directories
        data_dir = osp.join(data_path, 'RobotCar', scene)

        # decide which sequences to use
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.imgs = []
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)
            # read the image timestamps
            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]

            if real:  # poses from integration of VOs
                if vo_lib == 'stereo':
                    vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')
                    p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
                elif vo_lib == 'gps':
                    vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
                    p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
                else:
                    raise NotImplementedError
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'r') as f:
                    vo_stats[seq] = pickle.load(f)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            else:  # GT poses
                pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.imgs.extend([osp.join(seq_dir, 'stereo', 'centre_processed', '{:d}.png'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                              align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                              align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))

        # camera model and image loader (only use while pre_processing)
        camera_model = CameraModel('./data/robotcar_camera_models', osp.join('stereo', 'centre'))
        self.im_loader = partial(robotcar_loader, model=camera_model)

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                if self.undistort:
                    img = np.uint8(load_image(self.imgs[index], loader=self.im_loader))
                else:
                    img = load_image(self.imgs[index])
                pose = np.float32(self.poses[index])
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)

        return img, pose

    def __len__(self):
        return len(self.poses)

class MF(data.Dataset):
    def __init__(self, dataset, include_vos=False, no_duplicates=False, *args, **kwargs):

        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.real = kwargs.pop('real', False)
        self.include_vos = include_vos
        self.train = kwargs['train']
        self.vo_func = kwargs.pop('vo_func', calc_vos_simple)
        self.no_duplicates = no_duplicates

        if dataset == '7Scenes':
            self.dset = SevenScenes(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = SevenScenes(*args, skip_images=True, real=False, **kwargs)
        elif dataset == 'RobotCar':
            self.dset = RobotCar(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = RobotCar(*args, skip_images=True, real=False, **kwargs)
        else:
            raise NotImplementedError

        self.L = self.steps * self.skip

    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
        else:
            skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()
        offsets -= offsets[len(offsets) / 2]
        if self.no_duplicates:
            offsets += self.steps/2 * self.skip
        offsets = offsets.astype(np.int)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx = self.get_indices(index)
        clip = [self.dset[i] for i in idx]

        imgs  = torch.stack([c[0] for c in clip], dim=0)
        poses = torch.stack([c[1] for c in clip], dim=0)
        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0]
            if self.real:  # absolute poses need to come from the GT dataset
                clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
                poses = torch.stack([c[1] for c in clip], dim=0)
            poses = torch.cat((poses, vos), dim=0)

        return imgs, poses

    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L
