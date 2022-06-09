import numpy as np
import glob
import os
import cv2

# import matplotlib.pyplot as plt
import frame_utils

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict


# training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
#                     45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
#                     74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
#                     101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
#                     121, 122, 123, 124, 125, 126, 127, 128]
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def scale_operation(images, intrinsics, s):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    ht2 = int(s * ht1)
    wd2 = int(s * wd1)
    intrinsics[:, 0] *= s
    intrinsics[:, 1] *= s
    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)
    return images, intrinsics


def crop_operation(images, intrinsics, crop_h, crop_w):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    x0 = (wd1 - crop_w) // 2
    y0 = (ht1 - crop_h) // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    images = images[:, :, y0:y1, x0:x1]
    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0
    return images, intrinsics


def random_scale_and_crop(images, depths, masks, intrinsics, resize=[-1, -1], crop_size=[448, 576]):
    s = 2 ** np.random.uniform(-0.1, 0.4)

    ht1 = images.shape[2]
    wd1 = images.shape[3]
    if resize == [-1, -1]:
        ht2 = int(s * ht1)
        wd2 = int(s * wd1)
    else:
        ht2 = int(resize[0])
        wd2 = int(resize[1])

    intrinsics[:, 0] *= float(wd2) / wd1
    intrinsics[:, 1] *= float(ht2) / ht1

    depths = depths.unsqueeze(1)
    depths = F.interpolate(depths, [ht2, wd2], mode='nearest')
    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)

    x0 = np.random.randint(0, wd2 - crop_size[1] + 1)
    y0 = np.random.randint(0, ht2 - crop_size[0] + 1)
    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    images = images[:, :, y0:y1, x0:x1]
    depths = depths[:, :, y0:y1, x0:x1]
    depths = depths.squeeze(1)

    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0

    if masks is not None:
        masks = masks.unsqueeze(1)
        masks = F.interpolate(masks, [ht2, wd2], mode='nearest')
        masks = masks[:, :, y0:y1, x0:x1]
        masks = masks.squeeze(1)

    return images, depths, masks, intrinsics


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1 + 2 * n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i + 1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1 + 2 * n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j + 1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i // 2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


class DTUViewsynTrain(Dataset):
    def __init__(self, dataset_path, single, precomputed_depth_path, num_frames=2, light_number=-1, crop_size=[448, 576], resize=[-1, -1],
                 min_angle=3.0, max_angle=30.0, pairs_provided=False, foreground_mask_path=None, return_mask=False, target_views=None,
                 source_views=None, data_augmentation=True):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.light_number = light_number
        self.crop_size = crop_size
        self.resize = resize
        self.single = single
        self.precomputed_depth_path = precomputed_depth_path

        self.foreground_mask_path = foreground_mask_path
        self.return_mask = return_mask
        if return_mask:
            assert (foreground_mask_path is not None)

        # if not none, we assume source and target images (i.e. 1st-N element and the 0th element returned) are from two different sets
        if target_views is None:
            target_views = np.arange(49)
        if source_views is None:
            source_views = np.arange(49)

        self.target_views = target_views
        self.source_views = source_views

        self.data_augmentation = data_augmentation

        self._build_dataset_index()
        self._load_poses()
        self.pairs_provided = pairs_provided
        if pairs_provided:
            self.pair_list = load_pair(os.path.join(dataset_path, 'pair.txt'))

    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:, None], np.linalg.inv(poses[None, :]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR, axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        pose_glob = os.path.join(self.dataset_path, self.single, "cams", "*.txt")
        extrinsics_list, intrinsics_list = [], []
        for cam_file in sorted(glob.glob(pose_glob))[:49]:
            extrinsics = np.loadtxt(cam_file, skiprows=1, max_rows=4, dtype=np.float)
            intrinsics = np.loadtxt(cam_file, skiprows=7, max_rows=3, dtype=np.float)

            # intrinsics[0] /= 8
            # intrinsics[1] /= 8
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

        poses = np.stack(extrinsics_list, 0)
        # print(np.var(poses[:, :3, 3], axis=0))
        intrinsics = np.stack(intrinsics_list, 0)

        # compute angle between all pairs of poses
        thetas = self._theta_matrix(poses)

        self.poses = poses
        self.intrinsics = intrinsics

        self.pose_graph = []
        self.theta_list = []

        for i in range(len(poses)):
            cond = (thetas[i] > self.min_angle) & (thetas[i] < self.max_angle)
            ixs, = np.where(cond)
            ixs = np.intersect1d(ixs, self.source_views)  # must be in source view
            self.pose_graph.append(ixs)

        for i in range(len(poses)):
            list_i = []
            for j in range(len(poses)):
                if j in self.source_views and thetas[i, j] > self.min_angle:
                    list_i.append((thetas[i, j], j))
            list_i = sorted(list_i)
            self.theta_list.append(list_i)

        # recentered_pose = recenter_poses(self.poses.copy())

        # dist_to_world = self.poses[:, 0:3, -1] # N x 3
        # dist_to_world = np.linalg.norm(dist_to_world, axis=1) # N
        # print('average camera distance to world: ', np.mean(dist_to_world), dist_to_world)

    def _build_dataset_index(self):
        self.dataset_index = []
        self.target_index = []

        self.scale_between_image_depth = 1.0
        self.scenes = {}

        images = sorted(glob.glob(os.path.join(self.dataset_path, self.single, "images", "*.jpg")))
        depths = sorted(glob.glob(os.path.join(self.precomputed_depth_path, self.single, "depths", "*.pfm")))

        self.scenes[self.single] = (images, depths)
        self.dataset_index += [(self.single, i) for i in range(len(images))]
        self.target_index += [(self.single, i) for i in range(len(images)) if i in self.target_views]

        print('Dataset length:', len(self.target_index))

    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, index):
        # scene_id, ix1 = self.dataset_index[index]
        scene_id, ix1 = self.target_index[index]
        image_list, depth_list = self.scenes[scene_id]

        if self.num_frames == 1:
            indicies = [ix1]

        else:
            if self.pairs_provided:
                neighbors = [int(x) for x in self.pair_list[str(ix1)]['pair'][:self.num_frames - 1]]
            else:
                if len(self.pose_graph[ix1]) < self.num_frames - 1:
                    neighbors = np.random.choice([x[1] for x in self.theta_list[ix1]][:(self.num_frames - 1) * 2],
                                                 self.num_frames - 1, replace=False)
                else:
                    neighbors = np.random.choice(self.pose_graph[ix1], self.num_frames - 1, replace=False)

                neighbors = neighbors.tolist()

            indicies = [ix1] + neighbors

            assert np.all(np.in1d(neighbors, self.source_views))

        # sanity check
        assert ix1 in self.target_views


        # print(sorted(indicies))
        canonical_pose = self.poses[0]  # img0_T_world

        images, depths, poses, intrinsics, masks = [], [], [], [], []
        for i in indicies:
            image = frame_utils.read_gen(image_list[i])
            depth = frame_utils.read_gen(depth_list[i])

            # print(image_list[i], depth_list[i])

            # depth = cv2.resize(depth, [image.shape[1], image.shape[0]])

            if self.return_mask:
                img_name = os.path.basename(image_list[i]).split('_')[1]  # e.g. 002

                # our img_name ranges [1,49] while the masks [0,48], so do the trick here
                img_name = str((int(img_name) - 1)).zfill(3)

                scene_name = scene_id.split('_')[0]  # e.g. scan3
                mask_fn = os.path.join(self.foreground_mask_path, scene_name, 'mask', img_name + '.png')

                mask = frame_utils.read_gen(mask_fn) / 255.0
                if mask.ndim == 3:
                    mask = mask[..., 0]  # in case encoded as colorful image
            else:
                mask = None

            pose = self.poses[i]
            pose = pose @ np.linalg.inv(canonical_pose)
            calib = self.intrinsics[i]
            images.append(image)
            depths.append(depth)
            poses.append(pose)
            intrinsics.append(calib)
            masks.append(mask)

        images = np.stack(images, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)

        if self.return_mask:
            masks = np.stack(masks, 0).astype(np.float32)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        # depth_f = depths.reshape((-1,))
        # print(np.median(depth_f[depth_f > 0]))

        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()

        if self.data_augmentation:
            images, depths, masks, intrinsics = \
                random_scale_and_crop(images, depths, masks, intrinsics, self.resize, self.crop_size)

        if self.return_mask:
            return images, depths, masks, poses, intrinsics
        else:
            return images, depths, poses, intrinsics

    def get_blob_info(self, index):
        scene_id, frame_id = self.dataset_index[index]
        # print(scene_id)

        env_id = scene_id.split('_')[0]
        light_id = int(scene_id.split('_')[1])

        return light_id, env_id, frame_id

    def get_render_poses(self, radius=10):

        N_views = 60
        N_rots = 1

        pivot_dist = 800
        max_angle_x = radius
        max_angle_y = radius

        trans_t = lambda t: torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]]).float()

        rot_phi = lambda phi: torch.Tensor([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]]).float()

        rot_theta = lambda th: torch.Tensor([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).float()

        # def pose_spherical(theta, phi, radius):
        #     c2w = trans_t(radius)
        #     c2w = rot_phi(phi / 180. * np.pi) @ c2w
        #     c2w = rot_theta(theta / 180. * np.pi) @ c2w
        #     c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
        #     w2c = torch.linalg.inv(c2w)
        #     w2c = torch.Tensor(np.array([[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])) @ w2c
        #
        #     return w2c

        def pose_spherical(theta, phi, radius):
            w2c = rot_phi(phi / 180. * np.pi)
            w2c = rot_theta(theta / 180. * np.pi) @ w2c
            w2c = trans_t(radius) @ w2c

            return w2c

        # for i_batch, data_blob in enumerate(val_loader):
        #     if i_batch == 3:
        #         images, depths, poses, intrinsics = data_blob
        #         center_pose = poses[0, 0].cuda()  # 4x4
        #         break

        center_pose = torch.tensor(self.poses[23] @ np.linalg.inv(self.poses[0])).float().cuda() # 4 x 4, cam_T_world

        # center_pose = ref_poses[19] # 4 x 4, cam_T_world

        pivot_T_cam = trans_t(-pivot_dist).float().cuda()
        # render_pose_T_pivot = pose_spherical(0, 0, pivot_dist)
        render_poses_T_pivot = [pose_spherical(max_angle_x * np.sin(theta), max_angle_y * np.cos(theta), pivot_dist).cuda() for theta in np.linspace(0, 2 * np.pi * N_rots, N_views * N_rots + 1)[:-1]]

        render_poses = [render_pose_T_pivot @ pivot_T_cam @ center_pose for render_pose_T_pivot in render_poses_T_pivot]  # render_pose_T_world
        render_poses = torch.stack(render_poses, 0)  # N_views x 4 x 4

        return render_poses
