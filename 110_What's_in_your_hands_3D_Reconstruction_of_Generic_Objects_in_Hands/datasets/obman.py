# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from datasets.base_data import BaseData

import os
import os.path as osp
import pickle
import torch
import numpy as np
import tqdm
from PIL import Image
from nnutils import mesh_utils, geom_utils


class Obman(BaseData):
    def __init__(self, cfg, dataset: str, split='val', is_train=True,
                 data_dir='../data/', cache=None):
        data_dir = osp.join(data_dir, 'obman')
        super().__init__(cfg, 'obman', split, is_train, data_dir)
        self.cache = cache if cache is not None else self.cfg.DB.CACHE
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],
            'hTo': [],
            'cTh': [],
        }
        if '-' in dataset:
            self.cls = dataset.split('-')[-1]
        else:
            self.cls = ''

        self.cache_file = osp.join(self.data_dir, 'Cache', '%s_%s.pkl' % (dataset, self.split))
        self.cache_mesh = osp.join(self.data_dir, 'Cache', '%s_%s_mesh.pkl' % (dataset, self.split))

        self.shape_dir = os.path.join(self.cfg.DB.DIR, 'obmanboj', '{}', '{}', 'models', 'model_normalized.obj')
        self.image_dir = osp.join(self.data_dir, split, 'rgb', '{}.jpg')
        self.mask_dir = osp.join(self.data_dir, split, 'segms_plus', '{}.png')
        self.meta_dir = os.path.join(self.data_dir, split, 'meta_plus', '{}.pkl')
    
    def preload_anno(self, load_keys=[]):
        for key in load_keys:
            self.anno[key] = []
        
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!')
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:

            index_list = [line.strip() for line in open(osp.join(self.data_dir, '%s.txt' % self.split))]
            for i, index in enumerate(tqdm.tqdm(index_list)):
                dset = self.cfg.DB.NAME
                if 'mini' in dset and i >= int(dset.split('mini')[-1]):
                    break

                meta_path = self.meta_dir.format(index)
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)

                self.anno['index'].append(index)
                self.anno['cad_index'].append(osp.join(meta_info["class_id"], meta_info["sample_id"]))
                cTo = torch.FloatTensor([meta_info['cTo']])
                cTh = torch.FloatTensor([meta_info['cTh']])
                hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True)
                hTo = torch.matmul(hTc, cTo)

                self.anno['cTh'].append(cTh[0])
                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(meta_info['hA'])

            os.makedirs(osp.dirname(self.cache_file), exist_ok=True)
            print('save cache')
            pickle.dump(self.anno, open(self.cache_file, 'wb'))

        self.preload_mesh()

    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!')
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh')
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                cls, id = key.split('/')
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls, id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))

    def get_image(self, index):
        return Image.open(self.image_dir.format(index))   
    
    def get_cam(self, *args):
        f = 480 / 128
        p = 0
        cam_intr = torch.FloatTensor([
            [480, 0, 128],
            [0, 480, 128],
            [0, 0, 1]
        ])
        return cam_intr # torch.FloatTensor([f, f]), torch.FloatTensor([p, p])
    
    def get_bbox(self, idx):
        return torch.FloatTensor([0, 0, 256, 256])
        
    def get_obj_mask(self, index):
        """R channel"""
        mask_file = self.mask_dir.format(index)
        if osp.exists(mask_file):
            mask = np.array(Image.open(mask_file))[..., 0]
            mask = (mask > 0) * 255
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = np.ones_like(np.array(self.get_image(index))[..., 0])
            mask = (mask > 0) * 255
            mask = Image.fromarray(mask.astype(np.uint8))
        return mask

    def get_hand_mask(self, index):
        """B channel or from hA?"""
        mask_file = self.mask_dir.format(index)
        if osp.exists(mask_file):
            mask = np.array(Image.open(mask_file))[..., 0]
            mask = (mask > 0) * 255
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = np.ones_like(np.array(self.get_image(index))[..., 2])
            mask = (mask > 0) * 255
            mask = Image.fromarray(mask.astype(np.uint8))
        return mask
    
    def __getitem__(self, idx):
        idx = self.map[idx] if self.map is not None else idx
        sample = {key: self.anno[key][idx] for key in self.anno}
        sample['mesh'] = self.obj2mesh[sample['cad_index']]
        return sample
