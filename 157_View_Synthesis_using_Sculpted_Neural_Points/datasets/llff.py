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

class LLFF(Dataset):
    def __init__(self, dataset_path, num_frames=5, crop_size=[448, 576], resize=[-1, -1], min_angle=4.0, max_angle=10.0, source_views=None, data_augmentation=True, start=0, end=9999):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.crop_size = crop_size
        self.resize = resize

        self.data_augmentation = data_augmentation

        scene_name = dataset_path.split('/')[-1]
        # f = open("/u/zeyum/z/TNT_median_depth/%s.txt" % scene_name, "r")
        # self.med_depth = [float(line) for line in f]
        # f.close()

        self.total_num_views = len([f for f in sorted(os.listdir(os.path.join(dataset_path, "DTU_format", "images")))])

        if source_views is None:
            source_views = np.arange(self.total_num_views)

        self.source_views = source_views

        self._build_dataset_index()
        self._load_poses()
        self._rescale_depths()

        self.start = start
        self.end = end

    def _rescale_depths(self):
        depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        depth_list = sorted(glob.glob(depth_glob))

        depths = []

        for i in range(len(depth_list)):
            depth = frame_utils.read_gen(depth_list[i])
            depths.append(depth)

        depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        min_depth = np.min(depths[depths > 0.0])
        max_depth = np.max(depths)

        Dnear_original = 1.0 / min_depth
        Dnear_target = 0.9 * .0025

        self.depth_scale = Dnear_target / Dnear_original

        self.depths[self.depths > 0.0] = np.clip(self.depths[self.depths > 0.0], min_depth, max_depth)

        self.depths = self.depths / self.depth_scale
        self.poses[:, :3, 3] = self.poses[:, :3, 3] / self.depth_scale

        min_depth = np.min(self.depths[self.depths > 0.0])
        max_depth = np.max(self.depths)

        print('min/max disparity after scaling: %.4f/%.4f' % (1. / max_depth, 1. / min_depth))

    def _calc_depth_confidence(self):
        intrinsics_depth = self.intrinsics.copy()
        w2c = self.extrinsics.copy()
        c2w = []
        for pose in w2c:
            c2w.append(np.linalg.inv(pose))
        c2w = np.stack(c2w)

        conf = nerf_utils.cal_depth_confidences(self.depths, c2w, intrinsics_depth)
        self.depths_confidence = conf


    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:, None], np.linalg.inv(poses[None, :]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR, axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.rad2deg(np.arccos(cos_theta))



    def _load_poses(self):
        pose_glob = os.path.join(self.dataset_path, "DTU_format", "cameras.npz")
        camera_data = np.load(pose_glob)

        poses = camera_data['extrinsics'] # N x 4 x 4
        intrinsics = camera_data['intrinsics'] # 3 x 3

        assert (poses.shape[0] == self.total_num_views)

        intrinsics = np.repeat(intrinsics.reshape(1, 3, 3), self.total_num_views, axis=0)

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

    def _build_dataset_index(self):
        image_glob = os.path.join(self.dataset_path, "DTU_format", "images", "*.jpg")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths_4", "*.pfm")
        depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths_nerf", "*.pfm")

        self.image_list = sorted(glob.glob(image_glob))
        self.depth_list = sorted(glob.glob(depth_glob))

        print(self.depth_list)

        images = []
        depths = []

        for i in range(self.total_num_views):
            image = frame_utils.read_gen(self.image_list[i])
            depth = frame_utils.read_gen(self.depth_list[i])

            images.append(image)
            depths.append(depth)

        self.images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        self.depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        self.scale_between_image_depth = self.images.shape[1] / self.depths.shape[1]
        assert abs(self.images.shape[2] / self.depths.shape[2] - self.scale_between_image_depth) < 1e-2

    def __len__(self):
        # return len(self.image_list)
        # return len(self.image_list)
        return len(self.source_views)
        # return 1 # debug

    def __getitem__(self, ix1):
        ix1 = self.source_views[ix1]

        if ix1 < self.start or ix1 >= self.end: return []
        # randomly sample neighboring frame

        if len(self.pose_graph[ix1]) < self.num_frames - 1:
            # randomly sampled from all other views s.t. >= min_angles
            if len(self.theta_list[ix1]) >=  self.num_frames - 1: # enough views
                ix2 = np.random.choice([x[1] for x in self.theta_list[ix1]][:(self.num_frames - 1) * 2], self.num_frames - 1, replace=False)
            else: # no enough views
                ix2 = np.random.choice([x for x in self.source_views if ix1 != x], self.num_frames - 1, replace=False)
        else:
            ix2 = np.random.choice(self.pose_graph[ix1], self.num_frames - 1, replace=False)

        assert np.all(np.in1d(ix2, self.source_views))

        ix2 = ix2.tolist()

        indices = [ix1] + ix2
        # print("no no no")

        # print(indices)

        images, poses, depths, intrinsics = [], [], [], []
        for i in indices:
            # image = frame_utils.read_gen(self.image_list[i])
            # depth = frame_utils.read_gen(self.depth_list[i])
            image = self.images[i]
            depth = self.depths[i]

            pose = self.poses[i]
            calib = self.intrinsics[i].copy()
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)
            depths.append(depth)

        images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32) # N x H x W

        # if self.scale > 0:
        #     # depth_f = depths.reshape((-1,))
        #     # scale = 1000 #600 / np.median(depth_f[depth_f > 0])

        #     # depths *= scale
        #     scale = self.scale

        #     # scale = 2000 / self.med_depth[ix1]
        #     # print(scale)
        #     poses[:, :3, 3] *= scale

        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)  # N x 3 x H x W
        images = images.contiguous()

        if self.data_augmentation:
            images, depths, _, intrinsics = \
                random_scale_and_crop(images, depths, None, intrinsics, self.resize, self.crop_size)

        # for op, param in self.size_operations:
        #     if op == "scale":
        #         images, intrinsics = scale_operation(images, intrinsics, param)
        #     elif op == "crop":
        #         images, intrinsics = crop_operation(images, intrinsics, *param)

        return images, depths, poses, intrinsics


class LLFFTest(Dataset):
    def __init__(self, dataset_path, num_frames=5, size_operations=[], min_angle=4.0, max_angle=10.0, source_views=None, start=0, end=9999):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.size_operations = size_operations

        scene_name = dataset_path.split('/')[-1]
        # f = open("/u/zeyum/z/TNT_median_depth/%s.txt" % scene_name, "r")
        # self.med_depth = [float(line) for line in f]
        # f.close()

        self.total_num_views = len([f for f in sorted(os.listdir(os.path.join(dataset_path, "DTU_format", "images")))])

        if source_views is None:
            source_views = np.arange(self.total_num_views)

        self.source_views = source_views

        self._build_dataset_index()
        self._load_poses()
        self._rescale_depths()

        self.start = start
        self.end = end

    def _rescale_depths(self):
        depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        depth_list = sorted(glob.glob(depth_glob))

        depths = []

        for i in range(len(depth_list)):
            depth = frame_utils.read_gen(depth_list[i])
            depths.append(depth)

        depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        min_depth = np.min(depths[depths > 0.0])
        max_depth = np.max(depths)

        Dnear_original = 1.0 / min_depth
        Dnear_target = 0.9 * .0025

        self.depth_scale = Dnear_target / Dnear_original

        self.depths[self.depths > 0.0] = np.clip(self.depths[self.depths > 0.0], min_depth, max_depth)

        self.depths = self.depths / self.depth_scale
        self.poses[:, :3, 3] = self.poses[:, :3, 3] / self.depth_scale

        min_depth = np.min(self.depths[self.depths > 0.0])
        max_depth = np.max(self.depths)

        print('min/max disparity after scaling: %.4f/%.4f' % (1. / max_depth, 1. / min_depth))

    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:, None], np.linalg.inv(poses[None, :]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR, axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        pose_glob = os.path.join(self.dataset_path, "DTU_format", "cameras.npz")
        camera_data = np.load(pose_glob)

        poses = camera_data['extrinsics'] # N x 4 x 4
        intrinsics = camera_data['intrinsics'] # 3 x 3

        assert (poses.shape[0] == self.total_num_views)

        intrinsics = np.repeat(intrinsics.reshape(1, 3, 3), self.total_num_views, axis=0)

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

    def _build_dataset_index(self):
        image_glob = os.path.join(self.dataset_path, "DTU_format", "images", "*.jpg")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths_4", "*.pfm")
        depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths_nerf", "*.pfm")

        self.image_list = sorted(glob.glob(image_glob))
        self.depth_list = sorted(glob.glob(depth_glob))

        images = []
        depths = []

        for i in range(self.total_num_views):
            image = frame_utils.read_gen(self.image_list[i])
            depth = frame_utils.read_gen(self.depth_list[i])

            images.append(image)
            depths.append(depth)

        self.images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        self.depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

    def __len__(self):
        # return len(self.image_list)
        # return 2 # debug
        return len(self.source_views)
        # return 1 # debug

    def __getitem__(self, ix1):
        ix1 = self.source_views[ix1]

        if ix1 < self.start or ix1 >= self.end: return []
        # randomly sample neighboring frame

        if len(self.pose_graph[ix1]) < self.num_frames - 1:
            # randomly sampled from all other views s.t. >= min_angles
            if len(self.theta_list[ix1]) >= self.num_frames - 1:  # enough views
                ix2 = np.random.choice([x[1] for x in self.theta_list[ix1]][:(self.num_frames - 1) * 2],
                                       self.num_frames - 1, replace=False)
            else:  # no enough views
                ix2 = np.random.choice([x for x in self.source_views if ix1 != x], self.num_frames - 1, replace=False)
        else:
            ix2 = np.random.choice(self.pose_graph[ix1], self.num_frames - 1, replace=False)

        assert np.all(np.in1d(ix2, self.source_views))

        ix2 = ix2.tolist()

        indices = [ix1] + ix2
        # print("no no no")

        # print(indices)

        images, poses, depths, intrinsics = [], [], [], []
        for i in indices:
            # image = frame_utils.read_gen(self.image_list[i])
            # depth = frame_utils.read_gen(self.depth_list[i])
            image = self.images[i]
            depth = self.depths[i]

            pose = self.poses[i]
            calib = self.intrinsics[i].copy()
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)
            depths.append(depth)

        images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32) # N x H x W

        # if self.scale > 0:
        #     # depth_f = depths.reshape((-1,))
        #     # scale = 1000 #600 / np.median(depth_f[depth_f > 0])

        #     # depths *= scale
        #     scale = self.scale

        #     # scale = 2000 / self.med_depth[ix1]
        #     # print(scale)
        #     poses[:, :3, 3] *= scale

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)  # N x 3 x H x W
        images = images.contiguous()

        for op, param in self.size_operations:
            if op == "scale":
                images, intrinsics = scale_operation(images, intrinsics, param)
            elif op == "crop":
                images, intrinsics = crop_operation(images, intrinsics, *param)

        return images, depths, poses, intrinsics, 1.0

class LLFFViewsynTrain(Dataset):
    def __init__(self, dataset_path, single, precomputed_depth_path, num_frames=5, crop_size=[448, 576], resize=[-1, -1], min_angle=4.0, max_angle=10.0, source_views=None, data_augmentation=True, start=0, end=9999):
        self.dataset_path = os.path.join(dataset_path, single)

        self.precomputed_depth_path = os.path.join(precomputed_depth_path, single)

        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.crop_size = crop_size
        self.resize = resize

        self.data_augmentation = data_augmentation

        scene_name = dataset_path.split('/')[-1]
        # f = open("/u/zeyum/z/TNT_median_depth/%s.txt" % scene_name, "r")
        # self.med_depth = [float(line) for line in f]
        # f.close()

        self.total_num_views = len([f for f in sorted(os.listdir(os.path.join(self.dataset_path, "DTU_format", "images")))])

        if source_views is None:
            source_views = np.arange(self.total_num_views)

        self.source_views = source_views

        self._build_dataset_index()
        self._load_poses()
        self._rescale_depths()

        self.start = start
        self.end = end

    def _rescale_depths(self):
        # get the orginal depth file for scaling
        depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        depth_list = sorted(glob.glob(depth_glob))

        depths = []

        for i in range(len(depth_list)):
            depth = frame_utils.read_gen(depth_list[i])
            depths.append(depth)

        depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        min_depth = np.min(depths[depths > 0.0])
        max_depth = np.max(depths)

        Dnear_original = 1.0 / min_depth
        Dnear_target = 0.9 * .0025

        self.depth_scale = Dnear_target / Dnear_original

        # self.depths = np.clip(self.depths, min_depth, max_depth)

        self.depths = self.depths / self.depth_scale
        self.poses[:, :3, 3] = self.poses[:, :3, 3] / self.depth_scale

        min_depth = np.min(self.depths[self.depths > 0.0])
        max_depth = np.max(self.depths)

        print('min/max disparity after scaling: %.4f/%.4f' % (1. / max_depth, 1. / min_depth))


    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:, None], np.linalg.inv(poses[None, :]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR, axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        pose_glob = os.path.join(self.dataset_path, "DTU_format", "cameras.npz")
        camera_data = np.load(pose_glob)

        poses = camera_data['extrinsics'] # N x 4 x 4
        intrinsics = camera_data['intrinsics'] # 3 x 3

        # print(np.linalg.inv(poses[0])) # debug
        # assert False

        assert (poses.shape[0] == self.total_num_views)

        intrinsics = np.repeat(intrinsics.reshape(1, 3, 3), self.total_num_views, axis=0)

        self.poses = poses[np.array(self.source_views)]
        self.intrinsics = intrinsics[np.array(self.source_views)]


    def _build_dataset_index(self):
        image_glob = os.path.join(self.dataset_path, "DTU_format", "images", "*.jpg")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths_4", "*.pfm")
        # depth_glob = os.path.join(self.dataset_path, "DTU_format", "depths", "*.pfm")
        depth_glob = os.path.join(self.precomputed_depth_path, "depths", "*.pfm")

        self.image_list = sorted(glob.glob(image_glob))
        self.depth_list = sorted(glob.glob(depth_glob))

        self.image_list = list(map(self.image_list.__getitem__, self.source_views))
        self.depth_list = list(map(self.depth_list.__getitem__, self.source_views))

        # print(os.path.join(self.dataset_path, "DTU_format", "images", "*.jpg"))
        # print(self.image_list)

        images = []
        depths = []

        for i in range(len(self.source_views)):
            image = frame_utils.read_gen(self.image_list[i])
            depth = frame_utils.read_gen(self.depth_list[i])

            depth = cv2.resize(depth, (400, 300), interpolation=cv2.INTER_LINEAR)

            images.append(image)
            depths.append(depth)

        self.images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        self.depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        print('Dataset length:', len(self.image_list))

    def __len__(self):
        # return len(self.image_list)
        return len(self.image_list)
        # return 1 # debug

    def get_render_poses(self, radius=None):
        render_poses_raw = np.load(os.path.join(self.dataset_path, 'render_poses_raw.npy')) # 1 x 4 x 4

        # re-scale depth
        render_poses_raw[:, :3, 3] = render_poses_raw[:, :3, 3] / self.depth_scale

        return torch.tensor(render_poses_raw).float().cuda()

    def __getitem__(self, ix1):
        if ix1 < self.start or ix1 >= self.end: return []
        # randomly sample neighboring frame

        assert self.num_frames == 1

        if self.num_frames == 1:
            indices = [ix1]

        else:
            if len(self.pose_graph[ix1]) < self.num_frames - 1:
                # randomly sampled from all other views s.t. >= min_angles
                ix2 = np.random.choice([x[1] for x in self.theta_list[ix1]][:(self.num_frames - 1) * 2],
                                       self.num_frames - 1, replace=False)
            else:
                ix2 = np.random.choice(self.pose_graph[ix1], self.num_frames - 1, replace=False)

            assert np.all(np.in1d(ix2, self.source_views))

            ix2 = ix2.tolist()

            indices = [ix1] + ix2
        # print("no no no")

        # print(indices)

        images, poses, depths, intrinsics = [], [], [], []
        for i in indices:
            # image = frame_utils.read_gen(self.image_list[i])
            # depth = frame_utils.read_gen(self.depth_list[i])
            image = self.images[i]
            depth = self.depths[i]

            pose = self.poses[i]
            calib = self.intrinsics[i].copy()
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)
            depths.append(depth)

        images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32) # N x H x W

        # if self.scale > 0:
        #     # depth_f = depths.reshape((-1,))
        #     # scale = 1000 #600 / np.median(depth_f[depth_f > 0])

        #     # depths *= scale
        #     scale = self.scale

        #     # scale = 2000 / self.med_depth[ix1]
        #     # print(scale)
        #     poses[:, :3, 3] *= scale

        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)  # N x 3 x H x W
        images = images.contiguous()

        if self.data_augmentation:
            images, depths, _, intrinsics = \
                random_scale_and_crop(images, depths, None, intrinsics, self.resize, self.crop_size)

        # for op, param in self.size_operations:
        #     if op == "scale":
        #         images, intrinsics = scale_operation(images, intrinsics, param)
        #     elif op == "crop":
        #         images, intrinsics = crop_operation(images, intrinsics, *param)

        return images, depths, poses, intrinsics
