from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import trimesh
import scipy
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from data_processing import utils, read_landmarks3d

SEED = 777
np.random.seed(SEED)


def get_datasets():
    return {'TextureDataset': TextureDataset, 'GeometryDataset': GeometryDataset, 'GeometryDataset_Pose': GeometryDataset_Pose, 'GeometryDataset_Pose_FullShape': GeometryDataset_Pose_FullShape}


class IFNetDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.dataset = get_datasets()[cfg['dataset']]
        self.cfg = cfg
        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = cfg["training"]["num_workers"]

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = self.dataset("train", self.cfg)
            self.val_dataset = self.dataset("test", self.cfg)
        elif stage == "test":
            self.test_dataset = self.dataset("test", self.cfg)
        elif stage == "predict":
            self.predict_dataset = self.dataset("predict", self.cfg)
            # self.test_dataset = self.dataset("test", self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)


class IFNetDataset(Dataset):
    def __init__(self, mode, cfg):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TextureDataset(IFNetDataset):

    def __init__(self, mode, cfg):

        self.cfg = cfg
        self.path = cfg['data_path']
        self.mode = mode
        self.sigma_voxel = cfg['preprocessing']['voxelized_colored_pointcloud_sampling']['sigma']
        self.sigma_sampling = cfg['preprocessing']['color_sampling']['sigma']

        if mode == 'train':
            data_train = np.load(cfg['split_file'])['train']
            scores_train = np.load(cfg['split_file'])['train_score']

            data_val = np.load(cfg['split_file'])['val']
            scores_val = np.load(cfg['split_file'])['val_score']

            data_train_filtered = np.array(data_train)[np.where(
                np.array(scores_train) > cfg['overlap_score_threshold'])]
            data_val_filtered = np.array(data_val)[np.where(
                np.array(scores_val) > cfg['overlap_score_threshold'])]

            data_test = np.load(cfg['split_file'])['test']
            scores_test = np.load(cfg['split_file'])['test_score']

            data_train_filtered = np.array(data_train)[np.where(
                np.array(scores_train) > cfg['overlap_score_threshold'])]
            data_val_filtered = np.array(data_val)[np.where(
                np.array(scores_val) > cfg['overlap_score_threshold'])]

            data_test_filtered = np.array(data_test)[np.where(
                np.array(scores_test) > cfg['overlap_score_threshold'])]

            test_idx = np.random.randint(0, len(data_test_filtered), 100)
            index_test = [True] * len(data_test_filtered)
            index_test = np.array(index_test)
            index_test[test_idx] = False

            data_test_filtered = data_test_filtered[index_test]

            self.data = np.concatenate((data_train_filtered,
                                        data_val_filtered, data_test_filtered), axis=0)
            print(">>>>>>>>> With a threshold of {}, the number of {} data is filtered from {} to {}".format(
                cfg['overlap_score_threshold'], mode, len(data_train) + len(data_val) + len(data_test), len(self.data)))

        elif mode != 'predict':
            self.data = np.load(cfg['split_file'])[mode]
            scores = np.load(cfg['split_file'])[mode+'_score']
            self.data = np.array(self.data)[np.where(
                np.array(scores) > cfg['overlap_score_threshold'])]
            print(">>>>>>>>> With a threshold of {}, the number of {} data is filtered from {} to {}".format(
                cfg['overlap_score_threshold'], mode, len(scores), len(self.data)))

        else:
            self.data = np.load(cfg['split_file'])[mode]

        if mode == 'test':
            test_idx = np.random.randint(0, len(self.data), 100)
            # print("test_idx:", test_idx)
            self.data = self.data[test_idx]

        # self.data = np.load(cfg['split_file'])[mode]
        self.res = cfg['input_resolution']
        self.bbox_str = cfg['data_bounding_box_str']
        self.bbox = cfg['data_bounding_box']
        self.num_gt_rgb_samples = cfg['preprocessing']['color_sampling']['sample_number']
        self.ratio = cfg['preprocessing']['color_sampling']['ratio']

        self.sample_points_per_object = cfg['training']['sample_points_per_object']
        self.pointcloud_samples = cfg['input_points_number']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]

        path = os.path.normpath(path)
        challenge = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = os.path.splitext(path.split(os.sep)[-1])[0]

        if self.mode == 'predict':
            voxel_path = os.path.join(self.path, split, gt_file_name,
                                      '{}_voxelized_colored_point_cloud_res{}_points{}_bbox{}_predict.npz'
                                      .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))

            R = np.load(voxel_path)['R']
            G = np.load(voxel_path)['G']
            B = np.load(voxel_path)['B']
            S = np.load(voxel_path)['S']

            R = np.reshape(R, (self.res,)*3)
            G = np.reshape(G, (self.res,)*3)
            B = np.reshape(B, (self.res,)*3)
            S = np.reshape(S, (self.res,)*3)
            input = np.array([R, G, B, S])

            # path_surface = os.path.join(
            #     self.path, split, gt_file_name, gt_file_name + '_normalized.obj')
            path_surface = os.path.join('/itet-stor/leilil/net_scratch/if-net/3dv/experiments/IFNetGeometrySMPLGT_EarlyFusion/estimated_smpl_128_balanced_loss/geometry_reconstruction',
                                        gt_file_name, full_file_name + '_reconstruction.obj')
            if not os.path.exists(path_surface):
                print(path_surface, "not exist")
                # continue
            mesh = trimesh.load(path_surface)
            # create new uncolored mesh for color prediction
            pred_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
            # pred_mesh = pred_mesh.subdivide().subdivide()
            pred_verts_gird_coords = utils.to_grid_sample_coords(
                pred_mesh.vertices, self.bbox)

            return {'inputs': np.array(input, dtype=np.float32), 'path': path, 'mesh_path': path_surface, 'grid_coords': np.array(pred_verts_gird_coords, dtype=np.float32)}

        voxel_path = os.path.join(self.path, split, gt_file_name,
                                  '{}_voxelized_colored_point_cloud_res{}_points{}_bbox{}.npz'
                                  .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))

        R = np.load(voxel_path)['R']
        G = np.load(voxel_path)['G']
        B = np.load(voxel_path)['B']
        S = np.load(voxel_path)['S']

        R = np.reshape(R, (self.res,)*3)
        G = np.reshape(G, (self.res,)*3)
        B = np.reshape(B, (self.res,)*3)
        S = np.reshape(S, (self.res,)*3)
        input = np.array([R, G, B, S])

        rgb_samples_path = os.path.join(self.path, split[:-8], gt_file_name,
                                        '{}_normalized_color_samples{}_sigma{}_bbox{}.npz'
                                        .format(gt_file_name, self.num_gt_rgb_samples, self.sigma_sampling, self.bbox_str))

        rgb_samples_npz = np.load(rgb_samples_path)
        rgb_coords_sigma = rgb_samples_npz['grid_coords']
        rgb_values_sigma = rgb_samples_npz['colors']
        subsample_indices = np.random.randint(
            0, len(rgb_values_sigma), int(self.sample_points_per_object * self.ratio))
        rgb_coords_sigma = rgb_coords_sigma[subsample_indices]
        rgb_values_sigma = rgb_values_sigma[subsample_indices]

        rgb_samples_clean_path = os.path.join(self.path, split[:-8], gt_file_name,
                                              '{}_normalized_color_samples{}_bbox{}.npz'
                                              .format(gt_file_name, self.num_gt_rgb_samples, self.bbox_str))

        rgb_samples_npz_clean = np.load(rgb_samples_clean_path)
        rgb_coords_clean = rgb_samples_npz_clean['grid_coords']
        rgb_values_clean = rgb_samples_npz_clean['colors']

        subsample_indices_clean = np.random.randint(
            0, len(rgb_values_clean), self.sample_points_per_object - int(self.sample_points_per_object * self.ratio))
        rgb_coords_clean = rgb_coords_clean[subsample_indices_clean]
        rgb_values_clean = rgb_values_clean[subsample_indices_clean]

        rgb_coords = np.concatenate((
            rgb_coords_sigma, rgb_coords_clean), axis=0)
        rgb_values = np.concatenate((
            rgb_values_sigma, rgb_values_clean), axis=0)

        return {'grid_coords': np.array(rgb_coords, dtype=np.float32), 'rgb': np.array(rgb_values, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path': path}


class GeometryDataset(IFNetDataset):

    def __init__(self, mode, cfg):
        self.cfg = cfg
        self.path = cfg['data_path']
        self.mode = mode
        self.data = np.load(cfg['split_file'])[mode]
        if mode != 'predict':
            scores = np.load(cfg['split_file'])[mode+'_score']
            self.data = np.array(self.data)[np.where(
                np.array(scores) > cfg['overlap_score_threshold'])]
            print(">>>>>>>>> With a threshold of {}, the number of {} data is filtered from {} to {}".format(
                cfg['overlap_score_threshold'], mode, len(scores), len(self.data)))

        self.res = cfg['input_resolution']
        self.bbox_str = cfg['data_bounding_box_str']

        self.sample_distribution = np.array(
            cfg['preprocessing']['geometry_sampling']['sample_distribution'])
        self.sample_sigmas = cfg['preprocessing']['geometry_sampling']['sample_sigmas']

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.num_sample_points = cfg['training']['sample_points_per_object']
        self.pointcloud_samples = cfg['input_points_number']
        # compute number of samples per sampling method
        self.num_samples = np.rint(
            self.sample_distribution * self.num_sample_points).astype(np.uint32)

        bbox = cfg['data_bounding_box']
        resolution = cfg['generation']['retrieval_resolution']
        grid_points = utils.create_grid_points_from_xyz_bounds(
            *self.cfg['data_bounding_box'], resolution)
        grid_coords = utils.to_grid_sample_coords(grid_points, bbox)
        self.grid_coords = grid_coords.reshape([len(grid_points), 3])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        path = os.path.normpath(path)
        challenge = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = os.path.splitext(path.split(os.sep)[-1])[0]

        voxel_path = os.path.join(self.path, split, gt_file_name,
                                  '{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'
                                  .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))

        occupancies = np.unpackbits(
            np.load(voxel_path)['compressed_occupancies'])
        input = np.reshape(occupancies, (self.res,)*3)

        if self.mode == 'predict' or self.mode == 'test':
            return {'inputs': np.array(input, dtype=np.float32), 'path': path, 'grid_coords': np.array(self.grid_coords, dtype=np.float32)}
        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = os.path.join(self.path, split[:-8], gt_file_name,
                                                 '{}_normalized_boundary_{}_samples.npz'
                                                 .format(gt_file_name, self.sample_sigmas[i]))

            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(
                0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        return {'grid_coords': np.array(coords, dtype=np.float32), 'occupancies': np.array(occupancies, dtype=np.float32), 'points': np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path': path}


class GeometryDataset_Pose(IFNetDataset):

    def __init__(self, mode, cfg):
        self.cfg = cfg
        self.path = cfg['data_path']
        self.mode = mode

        if mode == 'train':
            data_train = np.load(cfg['split_file'])['train']
            scores_train = np.load(cfg['split_file'])['train_score']

            data_val = np.load(cfg['split_file'])['val']
            scores_val = np.load(cfg['split_file'])['val_score']

            data_train_filtered = np.array(data_train)[np.where(
                np.array(scores_train) > cfg['overlap_score_threshold'])]
            data_val_filtered = np.array(data_val)[np.where(
                np.array(scores_val) > cfg['overlap_score_threshold'])]

            data_test = np.load(cfg['split_file'])['test']
            scores_test = np.load(cfg['split_file'])['test_score']

            data_train_filtered = np.array(data_train)[np.where(
                np.array(scores_train) > cfg['overlap_score_threshold'])]
            data_val_filtered = np.array(data_val)[np.where(
                np.array(scores_val) > cfg['overlap_score_threshold'])]

            data_test_filtered = np.array(data_test)[np.where(
                np.array(scores_test) > cfg['overlap_score_threshold'])]

            test_idx = np.random.randint(0, len(data_test_filtered), 100)
            index_test = [True] * len(data_test_filtered)
            index_test = np.array(index_test)
            index_test[test_idx] = False

            data_test_filtered = data_test_filtered[index_test]

            self.data = np.concatenate((data_train_filtered,
                                        data_val_filtered, data_test_filtered), axis=0)
            print(">>>>>>>>> With a threshold of {}, the number of {} data is filtered from {} to {}".format(
                cfg['overlap_score_threshold'], mode, len(data_train) + len(data_val) + len(data_test), len(self.data)))

        elif mode != 'predict':
            self.data = np.load(cfg['split_file'])[mode]
            scores = np.load(cfg['split_file'])[mode+'_score']
            self.data = np.array(self.data)[np.where(
                np.array(scores) > cfg['overlap_score_threshold'])]
            print(">>>>>>>>> With a threshold of {}, the number of {} data is filtered from {} to {}".format(
                cfg['overlap_score_threshold'], mode, len(scores), len(self.data)))

        else:
            self.data = np.load(cfg['split_file'])[mode]

        if mode == 'test':
            test_idx = np.random.randint(0, len(self.data), 100)
            # print("test_idx:", test_idx)
            self.data = self.data[test_idx]

        self.refine_with_estimated_smpl = cfg['refine_with_estimated_smpl']
        print('refine_with_estimated_smpl:', self.refine_with_estimated_smpl)

        # self.data = self.data[:10]
        self.res = cfg['input_resolution']
        self.bbox_str = cfg['data_bounding_box_str']
        # self.smpl = 'IFNetGeometrySMPLGT' in cfg['model']

        self.sample_distribution = np.array(
            cfg['preprocessing']['geometry_sampling']['sample_distribution'])
        self.sample_sigmas = cfg['preprocessing']['geometry_sampling']['sample_sigmas']

        print('sample_sigmas:', self.sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.num_sample_points = cfg['training']['sample_points_per_object']
        self.pointcloud_samples = cfg['input_points_number']
        # compute number of samples per sampling method
        self.num_samples = np.rint(
            self.sample_distribution * self.num_sample_points).astype(np.uint32)

        bbox = cfg['data_bounding_box']
        resolution = cfg['generation']['retrieval_resolution']
        grid_points = utils.create_grid_points_from_xyz_bounds(
            *self.cfg['data_bounding_box'], resolution)
        grid_coords = utils.to_grid_sample_coords(grid_points, bbox)
        self.grid_coords = grid_coords.reshape([len(grid_points), 3])
        self.mapping = [8, 12, 9, 13, 10, 21,
                        24, 20, 23, 1, 0, 5, 2, 6, 3, 7, 4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        path = os.path.normpath(path)
        challenge = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = os.path.splitext(path.split(os.sep)[-1])[0]

        if self.refine_with_estimated_smpl and self.mode != 'predict':
            voxel_path = os.path.join(self.path, split, gt_file_name,
                                      '{}_voxelized_point_cloud_res{}_points{}_bbox{}_smpl_estimated.npz'
                                      .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))
        else:

            voxel_path = os.path.join(self.path, split, gt_file_name,
                                      '{}_voxelized_point_cloud_res{}_points{}_bbox{}_smpl.npz'
                                      .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))
        # else:
        #     voxel_path = os.path.join(self.path, split, gt_file_name,
        #                               '{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'
        #                               .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))

        # use standard
        # 170410-005-m-tt3n-a65a-low-res-result_normalized
        # voxel_path = "170410-005-m-tt3n-a65a-low-res-result_normalized_voxelized_point_cloud_res128_points100000_bbox-0.8,0.8,-0.15,2.1,-0.8,0.8.npz"

        occupancies = np.unpackbits(
            np.load(voxel_path)['compressed_occupancies'])
        input = np.reshape(occupancies, (self.res,)*3)

        smpl_occupancies = np.unpackbits(
            np.load(voxel_path)['smpl_compressed_occupancies'])
        smpl_input = np.reshape(smpl_occupancies, (self.res,)*3)

        if self.mode == 'predict' or self.mode == 'test':
            return {'inputs': np.array(input, dtype=np.float32), 'smpl_inputs': np.array(smpl_input, dtype=np.float32), 'path': path, 'grid_coords': np.array(self.grid_coords, dtype=np.float32)}

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = os.path.join(self.path, split[:-8], gt_file_name,
                                                 '{}_normalized_boundary_{}_samples.npz'
                                                 .format(gt_file_name, self.sample_sigmas[i]))

            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(
                0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        occ_dt = scipy.ndimage.distance_transform_edt(1-input)
        occ_dt = torch.Tensor(occ_dt).unsqueeze(0).unsqueeze(0)
        coords_dt = torch.Tensor(coords).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dt = F.grid_sample(occ_dt, coords_dt, padding_mode='border')
        dt = dt.flatten().numpy()

        # hard coded
        dt_mask = dt >= 16
        landmarks3d_path = os.path.join(
            self.path, split[:-8], gt_file_name, 'landmarks3d.txt')
        landmarks3d = read_landmarks3d.read(
            landmarks3d_path, self.cfg["data_bounding_box"])[self.mapping]
        landmarks3d = np.nan_to_num(landmarks3d)

        return {'grid_coords': np.array(coords, dtype=np.float32), 'occupancies': np.array(occupancies, dtype=np.float32), 'points': np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'smpl_inputs': np.array(smpl_input, dtype=np.float32), 'landmarks3d': np.array(landmarks3d, dtype=np.float32), 'path': path, 'dt_mask': np.array(dt_mask, dtype=np.uint8)}


class GeometryDataset_Pose_FullShape(IFNetDataset):

    def __init__(self, mode, cfg):
        self.cfg = cfg
        self.path = cfg['data_path']
        self.mode = mode
        self.data = np.load(cfg['split_file'])[mode]
        # if mode == 'test':
        #     self.data.sort()
        #     self.data = self.data[:10]
        #     print(self.data)
        self.res = cfg['input_resolution']
        self.bbox_str = cfg['data_bounding_box_str']

        self.sample_distribution = np.array(
            cfg['preprocessing']['geometry_sampling']['sample_distribution'])
        self.sample_sigmas = cfg['preprocessing']['geometry_sampling']['sample_sigmas']

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.num_sample_points = cfg['training']['sample_points_per_object']
        self.pointcloud_samples = cfg['input_points_number']
        # compute number of samples per sampling method
        self.num_samples = np.rint(
            self.sample_distribution * self.num_sample_points).astype(np.uint32)

        bbox = cfg['data_bounding_box']
        resolution = cfg['generation']['retrieval_resolution']
        grid_points = utils.create_grid_points_from_xyz_bounds(
            *self.cfg['data_bounding_box'], resolution)
        grid_coords = utils.to_grid_sample_coords(grid_points, bbox)
        self.grid_coords = grid_coords.reshape([len(grid_points), 3])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        path = os.path.normpath(path)
        challenge = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = os.path.splitext(path.split(os.sep)[-1])[0]

        voxel_path = os.path.join(self.path, split, gt_file_name,
                                  '{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'
                                  .format(full_file_name, self.res, self.pointcloud_samples, self.bbox_str))

        # use standard
        # 170410-005-m-tt3n-a65a-low-res-result_normalized
        voxel_path_full_shape = "170410-005-m-tt3n-a65a-low-res-result_normalized_voxelized_point_cloud_res128_points100000_bbox-0.8,0.8,-0.15,2.1,-0.8,0.8.npz"

        occupancies = np.unpackbits(
            np.load(voxel_path)['compressed_occupancies'])
        input = np.reshape(occupancies, (self.res,)*3)

        if self.cfg['data']["use_smpl"]:
            smpl_occupancies = np.unpackbits(
                np.load(voxel_path)['smpl_compressed_occupancies'])
            smpl_input = np.reshape(smpl_occupancies, (self.res,)*3)
        if self.cfg['data']['use_pose']:
            landmarks3d_path = os.path.join(
                self.path, split[:-8], gt_file_name, 'landmarks3d.txt')
            landmarks3d = read_landmarks3d.read(
                landmarks3d_path, self.cfg["data_bounding_box"])

        if self.mode == 'predict' or self.mode == 'test':
            out = {'inputs': np.array(input, dtype=np.float32), 'path': path, 'grid_coords': np.array(
                self.grid_coords, dtype=np.float32)}
            if self.cfg['data']['use_smpl']:
                out['smpl_inputs'] = np.array(smpl_input, dtype=np.float32)
            if self.cfg['data']['use_pose']:
                out['landmarks3d'] = np.array(landmarks3d, dtype=np.float32)
            return out

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            if self.cfg['data']['use_sdf']:
                boundary_samples_path = os.path.join(self.path, split[:-8], gt_file_name,
                                                     '{}_normalized_boundary_{}_sdf_samples.npz'
                                                     .format(gt_file_name, self.sample_sigmas[i]))

                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_coords = boundary_samples_npz['grid_coords']
                boundary_sample_occupancies = boundary_samples_npz['sdf']
                subsample_indices = np.random.randint(
                    0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                coords.extend(boundary_sample_coords[subsample_indices])
                occupancies.extend(
                    boundary_sample_occupancies[subsample_indices])
            else:
                boundary_samples_path = os.path.join(self.path, split[:-8], gt_file_name,
                                                     '{}_normalized_boundary_{}_samples.npz'
                                                     .format(gt_file_name, self.sample_sigmas[i]))

                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_coords = boundary_samples_npz['grid_coords']
                boundary_sample_occupancies = boundary_samples_npz['occupancies']
                subsample_indices = np.random.randint(
                    0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                coords.extend(boundary_sample_coords[subsample_indices])
                occupancies.extend(
                    boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        occ_dt = scipy.ndimage.distance_transform_edt(1-input)
        occ_dt = torch.Tensor(occ_dt).unsqueeze(0).unsqueeze(0)
        coords_dt = torch.Tensor(coords).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dt = F.grid_sample(occ_dt, coords_dt, padding_mode='border')
        dt = dt.flatten().numpy()

        # hard coded
        dt_mask = dt >= 16

        out = {'grid_coords': np.array(coords, dtype=np.float32), 'points': np.array(points, dtype=np.float32), 'inputs': np.array(
            input, dtype=np.float32), 'dt_mask': np.array(dt_mask, dtype=np.uint8), 'path': path}
        if self.cfg['data']['use_sdf']:
            out['sdf'] = np.array(occupancies, dtype=np.float32)
        else:
            out['occupancies'] = np.array(occupancies, dtype=np.float32)

        if self.cfg['data']['use_smpl']:
            out['smpl_inputs'] = np.array(smpl_input, dtype=np.float32)
        if self.cfg['data']['use_pose']:
            out['landmarks3d'] = np.array(landmarks3d, dtype=np.float32)

        return out
