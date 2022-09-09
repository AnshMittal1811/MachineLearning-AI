import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import auxiliary.my_utils as my_utils
import pickle
from os.path import join, dirname, exists
from easydict import EasyDict
import json
import csv
from termcolor import colored
import dataset.pointcloud_processor as pointcloud_processor
from copy import deepcopy

class ShapeNet(data.Dataset):
    """
    Shapenet Dataloader
    Uses Shapenet V1
    Make sure to respect shapenet Licence.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt, category, subcategory, svr=False, train=True):
        self.opt = opt
        self.num_sample = opt.number_points if train else 2500

        self.SVR = svr
        self.train = train
        self.mode = 'training' if train else 'validation'
        self.category = category
        self.subcategory = subcategory

        # Initialize pointcloud normalization functions
        self.init_normalization()
        self.init_singleview()

        if not opt.demo or not opt.use_default_demo_samples:
            my_utils.red_print('Create Shapenet Dataset...')
            # Define core path array
            self.dataset_path = os.path.join(opt.data_dir, 'ShapeNet')
            self.pointcloud_path = os.path.join(self.dataset_path, 'ShapeNetV1PointCloud')
            self.image_path = os.path.join(self.dataset_path, 'ShapeNetV1Renderings')

            # Create Cache path
            self.path_dataset = os.path.join(self.dataset_path, 'cache')
            if not os.path.exists(self.path_dataset):
                os.makedirs(self.path_dataset)
            self.path_dataset = os.path.join(self.path_dataset, '_'.join((self.opt.normalization, self.mode)))
            subcategory_valid = ''.join([s if s != '.' else ' ' for s in self.subcategory])
            self.cache_file = self.path_dataset + subcategory_valid + '_info.pkl'

            if not exists(self.image_path):
                os.system("chmod +x dataset/download_shapenet_renderings.sh")
                os.system("./dataset/download_shapenet_renderings.sh")

            self.num_image_per_object = 24
            self.idx_image_val = 0

            if not os.path.exists(self.cache_file):
                # Load taxonomy file
                self.taxonomy_path = os.path.join(opt.data_dir, 'ShapeNet', 'taxonomy.json')
                if not exists(self.taxonomy_path):
                    os.system("chmod +x dataset/download_shapenet_pointclouds.sh")
                    os.system("./dataset/download_shapenet_pointclouds.sh")

                # Load classes
                self.classes = [x for x in next(os.walk(self.pointcloud_path))[1]]
                with open(self.taxonomy_path, 'r') as f:
                    self.taxonomy = json.load(f)

                self.id2names = {}
                self.id2children = {}
                self.names2id = {}
                for dict_class in self.taxonomy:
                    if dict_class['synsetId'] in self.classes:
                        # name = dict_class['name'].split(sep=',')[0]
                        name = dict_class['name']
                        self.id2children[dict_class['synsetId']] = dict_class['children']
                        self.id2names[dict_class['synsetId']] = name
                        self.names2id[name] = dict_class['synsetId']

                self.category_id = self.names2id[category]
                self.childrenid2names = {}
                self.names2childrenid = {}
                for dict_class in self.taxonomy:
                    if dict_class['synsetId'] in self.id2children[self.category_id]:
                        # name = dict_class['name'].split(sep=',')[0]
                        name = dict_class['name']
                        self.childrenid2names[dict_class['synsetId']] = name
                        self.names2childrenid[name] = dict_class['synsetId']

                # Select class
                self.subcategory_id = self.names2childrenid[subcategory]

                # Load csv file
                self.csv_path = os.path.join(self.dataset_path, 'all.csv')
                if not exists(self.taxonomy_path):
                    raise ValueError(f'{self.taxonomy_path} does not exist.')

                # Compile list of pointcloud path by selected category
                dir_pointcloud = os.path.join(self.pointcloud_path, self.category_id)
                dir_image = os.path.join(self.image_path, self.category_id)
                list_pointcloud = []
                with open(self.csv_path) as csvfile:
                    self.csv_taxonomy = csv.DictReader(csvfile)
                    for dict_class in self.csv_taxonomy:
                        if (dict_class['synsetId'] == self.category_id and
                                dict_class['subSynsetId'] in self.subcategory_id):
                            pointcloud = dict_class['modelId'] + '.points.ply.npy'
                            list_pointcloud.append(pointcloud)

                if self.train:
                    list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 0.8)]
                else:
                    list_pointcloud = list_pointcloud[int(len(list_pointcloud) * 0.8):]

                print(
                    '    subcategory '
                    + colored(self.names2childrenid[self.subcategory], 'yellow')
                    + '  '
                    + colored(self.subcategory, 'cyan')
                    + ' Number Files: '
                    + colored(str(len(list_pointcloud)), 'yellow')
                )

                if len(list_pointcloud) != 0:
                    self.datapath = []
                    for pointcloud in list_pointcloud:
                        pointcloud_path = os.path.join(dir_pointcloud, pointcloud)
                        image_path = os.path.join(dir_image, pointcloud.split(".")[0], "rendering")
                        if not self.SVR or exists(image_path):
                            self.datapath.append((pointcloud_path, image_path, pointcloud, self.subcategory))
                        else:
                            my_utils.red_print(f"Rendering not found : {image_path}")

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

            self.data_metadata = [{'pointcloud_path': data[1], 'image_path': data[2], 'name': data[3],
                                   'subcategory': data[4]} for data in self.datas]

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

    def init_singleview(self):
        ## Define Image Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])

        self.validating = transforms.Compose([
            transforms.CenterCrop(127),
        ])

    def _getitem(self, index):

        pointcloud_path, image_path, pointcloud, subcategory = self.datapath[index]
        points = self.load(pointcloud_path)['points'][0]
        points[:, :3] = self.normalization_function(points[:, :3])
        return points.unsqueeze(0), pointcloud_path, image_path, pointcloud, subcategory

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
                       'image_path': return_dict['image_path'],
                       'subcategory': return_dict['subcategory']}

        # Image processing
        if self.SVR:  # TODO(msegu): maybe let self.SVR an input flag so that one dataset will be for svr and the other not
            if self.train:
                N = np.random.randint(1, self.num_image_per_object)
                im = Image.open(join(return_dict['image_path'], ShapeNet.int2str(N) + ".png"))
                im = self.dataAugmentation(im)  # random crop
            else:
                im = Image.open(join(return_dict['image_path'], ShapeNet.int2str(self.idx_image_val) + ".png"))
                im = self.validating(im)  # center crop
            im = self.transforms(im)  # scale
            im = im[:3, :, :]
            return_dict['image'] = im

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
            return self.load_image(path)

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

    def load_image(self, path):
        im = Image.open(path)
        im = self.validating(im)
        im = self.transforms(im)
        im = im[:3, :, :]
        return_dict = {
            'image': im.unsqueeze_(0),
            'operation': None,
            'path': path,
        }
        return return_dict


if __name__ == '__main__':
    print('Testing Shapenet dataset')
    opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
           "shapenet13": True}
    d = ShapeNet(EasyDict(opt), train=False, keep_track=True)
    print(d[1])
    a = len(d)
