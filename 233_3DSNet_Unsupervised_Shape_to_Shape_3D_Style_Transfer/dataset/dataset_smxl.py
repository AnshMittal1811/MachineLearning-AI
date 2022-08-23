from copy import deepcopy
from easydict import EasyDict
import numpy as np
import os
import pickle
from termcolor import colored
import torch
import torch.utils.data as data

import auxiliary.my_utils as my_utils
import dataset.pointcloud_processor as pointcloud_processor


class SMXL(data.Dataset):
    """
    SMXL Dataloader
    Uses SMAL V1, SMPL and SMIL
    Make sure to respect SMAL, SMPL and SMIL Licence.
    """

    def __init__(self, opt, unused_category, subcategory, unused_svr=False, train=True):
        self.opt = opt
        self.num_sample = opt.number_points if train else 2500

        self.train = train
        self.mode = 'training' if train else 'validation'
        self.subcategory = subcategory

        self.id2names = {0: 'cats', 1: 'dogs', 2: 'horses', 3: 'cows', 4: 'hippos', 5: 'male', 6: 'female'}
        self.names2id = {'cats': 0, 'dogs': 1, 'horses': 2, 'cows': 3, 'hippos': 4, 'male': 5, 'female': 6}

        # Initialize pointcloud normalization functions
        self.init_normalization()

        if not opt.demo or not opt.use_default_demo_samples:
            if len(opt.class_choice) > 0 and len(opt.class_choice) == 2:
                print('Initializing {} dataset for class {}.'.format(
                    self.mode, subcategory))
            else:
                raise ValueError('Argument class_choice must contain exactly two classes.')

            my_utils.red_print('Create SMXL Dataset...')
            # Define core path array
            self.dataset_path = os.path.join(opt.data_dir, 'SMXL')

            # Create Cache path
            self.path_dataset = os.path.join(self.dataset_path, 'cache')
            if not os.path.exists(self.path_dataset):
                os.makedirs(self.path_dataset)
            self.path_dataset = os.path.join(self.path_dataset, '_'.join((self.opt.normalization, self.mode)))
            self.cache_file = self.path_dataset + self.subcategory + '_info.pkl'

            if not os.path.exists(self.cache_file):
                # Compile list of pointcloud path by selected categories
                dir_pointcloud = os.path.join(self.dataset_path, self.subcategory, self.mode)
                list_pointcloud = sorted(os.listdir(dir_pointcloud))
                print(
                    '    subcategory '
                    + colored(self.names2id[self.subcategory], 'yellow')
                    + '  '
                    + colored(self.subcategory, 'cyan')
                    + ' Number Files: '
                    + colored(str(len(list_pointcloud)), 'yellow')
                )

                if len(list_pointcloud) != 0:
                    self.datapath = []
                    for pointcloud in list_pointcloud:
                        pointcloud_path = os.path.join(dir_pointcloud, pointcloud)
                        self.datapath.append((pointcloud_path, pointcloud, self.subcategory))

            # Preprocess and cache files
            self.preprocess()

    def preprocess(self):
        if os.path.exists(self.cache_file):
            # Reload dataset
            my_utils.red_print('Reload dataset : {}'.format(self.cache_file))
            with open(self.cache_file, 'rb') as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + self.subcategory + '_points.pth')
        else:
            # Preprocess dataset and put in cache for future fast reload
            my_utils.red_print('Preprocess dataset...')
            self.datas = [self._getitem(i) for i in range(len(self.datapath))]

            # Concatenate all processed files
            self.data_points = [data[0] for data in self.datas]
            # TODO(msegu): consider adding option to randomly select num_samples if we want to train with less samples
            self.data_points = torch.cat(self.data_points, 0)

            self.data_metadata = [{'pointcloud_path': data[1], 'name': data[2], 'subcategory': data[3]}
                                  for data in self.datas]

            # Save in cache
            with open(self.cache_file, 'wb') as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + self.subcategory + '_points.pth')

        my_utils.red_print('Dataset Size: ' + str(len(self.data_metadata)))

    def init_normalization(self):
        if not self.opt.demo:
            my_utils.red_print('Dataset normalization : ' + self.opt.normalization)

        if self.opt.normalization == 'UnitBall':
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == 'BoundingBox':
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def _getitem(self, index):

        pointcloud_path, pointcloud, subcategory = self.datapath[index]
        points = self.load(pointcloud_path)['points'][0]
        points[:, :3] = self.normalization_function(points[:, :3])
        return points.unsqueeze(0), pointcloud_path, pointcloud, subcategory

    def __getitem__(self, index):
        return_dict = deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        if self.opt.sample:
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]
        points = points[:, :3].contiguous()

        return_dict = {'points': points,
                       'pointcloud_path': return_dict['pointcloud_path'],
                       'subcategory': return_dict['subcategory']}
        return return_dict

    def __len__(self):
        return len(self.data_metadata)

    @staticmethod
    def int2str(N):
        if N < 10:
            return '0' + str(N)
        else:
            return str(N)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext == 'npy' or ext == 'ply' or ext == 'obj':
            return self.load_point_input(path)
        else:
            raise IOError("File extension .{} not supported. Must be one of '.npy', '.ply' or '.obj'.".format(ext))

    def load_point_input(self, path):
        ext = path.split('.')[-1]
        if ext == 'npy':
            points = np.load(path)
        elif ext == 'ply' or ext == 'obj':
            import pymesh
            points = pymesh.load_mesh(path).vertices
        else:
            print('invalid file extension')

        points = torch.from_numpy(points.copy()).float()
        operation = pointcloud_processor.Normalization(points, keep_track=True)
        if self.opt.normalization == 'UnitBall':
            operation.normalize_unitL2ball()
        elif self.opt.normalization == 'BoundingBox':
            operation.normalize_bounding_box()
        else:
            pass
        return_dict = {
            'points': points,
            'operation': operation,
            'path': path,
        }
        return return_dict


if __name__ == '__main__':
    print('Testing SMXL dataset')
    opt = EasyDict({'normalization': 'UnitBall', 'class_choice': ['cats', 'male'], 'sample': True, 'npoints': 2500,
                    'num_epochs': 5})
    dataset_a = SMXL(opt, subcategory=opt.class_choice[0], train=False)
    dataset_b = SMXL(opt, subcategory=opt.class_choice[1], train=False)

    print(dataset_a[1])
    a = len(dataset_a)
    b = len(dataset_b)

    # Check that random pairwise loading works as expected
    dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=1, shuffle=True)

    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=1, shuffle=True)

    for epoch in range(opt.num_epochs):
        for i, (data_a, data_b) in enumerate(zip(dataloader_a, dataloader_b)):
            if i == 2: break
            data_a = EasyDict(data_a)
            data_b = EasyDict(data_b)
            print(data_a.pointcloud_path, data_a.pointcloud_path)










