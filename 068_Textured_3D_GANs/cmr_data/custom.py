import os
import os.path as osp
import numpy as np

import scipy.io as sio
import copy

import torch
from torch.utils.data import Dataset

from . import base as base_data
from data.definitions import vg3k_class_set

import pycocotools.mask as mask_util
import cmr_data.image_utils as image_utils
import cmr_data.transformations as transformations
from skimage.io import imread

class CustomDataset(base_data.BaseDataset):
    def __init__(self, is_train, img_size, dataset, poses_dir=None,
                 unfiltered=False, enable_seg=False, add_flipped=False, rasterize_argmax=False,
                 semi_fraction=0.1):
        super().__init__(is_train, img_size)
        
        path = f'cache/{dataset}/detections.npy'
        self.detections = np.load(path, allow_pickle=True)
        
        self.kp_perm = [0]
        self.enable_seg = enable_seg
        self.add_flipped = add_flipped
        self.rasterize_argmax = rasterize_argmax
        if add_flipped:
            assert not is_train
        
        self.unfiltered = unfiltered
        if not unfiltered and poses_dir is not None:
            self.poses = torch.load(poses_dir)
            self.detections = self.detections[self.poses['indices']] # Pre-filter
        else:
            self.poses = None
        self.num_imgs = len(self.detections)
        
        # Filter and remap parts
        thresh_frequency = 0.25
        part_ids = set()
        for record in self.detections:
            filtered_parts = [x for x in record['parts'] if x['frequency'] >= thresh_frequency]
            record['parts'] = filtered_parts
            record['num_parts'] = len(filtered_parts)
            part_ids.update([x['class_id'] for x in filtered_parts])

        part_ids = sorted(part_ids)
        part_id_remapper = {x: y for x, y in zip(part_ids, range(len(part_ids)))}
        self.part_ids = part_ids
        self.part_id_remapper = part_id_remapper

        for record in self.detections:
            for part in record['parts']:
                part['class_id'] = part_id_remapper[part['class_id']]
        self.num_parts = len(part_ids)
        
        # Do topk selection for semi-supervision
        if not unfiltered and self.poses is not None:
            print('Semi-supervision:')
            nt = self.poses['w'].shape[-1]
            semi_w = self.poses['w']
            all_iou = self.poses['iou']
            semi_indices = []
            for k in range(nt):
                valid_k = semi_w.argmax(dim=-1) == k
                num_img = int(valid_k.sum().item() * semi_fraction)
                print(f'[{k}] {num_img}/{valid_k.sum().item()}')
                values, indices = (all_iou.max(dim=1).values * valid_k.float()).topk(num_img)
                semi_indices.append(indices[values > 0])
            semi_indices = torch.cat(semi_indices)
            semi_mask = torch.zeros(self.num_imgs)
            semi_mask[semi_indices] = 1
            self.semi_mask = semi_mask
            print(f'[total] {len(semi_indices)}/{self.num_imgs}')
        
        
        self.extra_img_keys = []
        if isinstance(img_size, list):
            for res in img_size[1:]:
                self.extra_img_keys.append(f'img_{res}')
                
        if not unfiltered and self.poses is None:
            # In pose estimation mode, load ground-truth rotations wherever available (only for evaluation purposes!)
            if dataset == 'cub':
                anno_path = osp.join('datasets/cub/data', 'train_cub_cleaned.mat')
                anno_sfm_path = osp.join('datasets/cub/sfm', 'anno_train.mat')
                self.gt_available = True
            elif ('car' in dataset or 'airplane' in dataset) and ('p3d' in dataset or 'imagenet' in dataset):
                p3d_class = dataset.split('_')[-1].replace('airplane', 'aeroplane')
                anno_path = osp.join('datasets/p3d/data', f'{p3d_class}_train.mat')
                anno_sfm_path = osp.join('datasets/p3d/sfm', f'{p3d_class}_train.mat')
                self.gt_available = True
            else:
                self.gt_available = False

            if self.gt_available:
                # Build index of paths
                path_index = {}
                for i, item in enumerate(self.detections):
                    p = osp.basename(item['image_path'].replace('\\', '/')) # Use filename as key
                    path_index[p] = i

                anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
                anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
                self.gt = {}
                for im, sfm in zip(anno, anno_sfm):
                    p = osp.basename(im.rel_path.replace('\\', '/'))
                    if p in path_index:
                        self.gt[path_index[p]] = sfm
        else:
            self.gt_available = False
    
    
    def __len__(self):
        if self.add_flipped:
            return 2*self.num_imgs
        else:
            return self.num_imgs
    
    def rasterize_seg(self, item):
        out = np.zeros((2 + self.num_parts, item['image_height'], item['image_width']), dtype=np.float32)
        global_mask = mask_util.decode(item['mask']).astype(np.bool)
        out[0, ~global_mask] = 1e-8 # Background
        out[1, global_mask] = 1e-8 # Foreground
        for i, s in enumerate(item['parts']):
            mask = mask_util.decode(s['mask']).astype(np.bool)
            out[s['class_id'] + 2, mask] = s['score']

        # Normalize
        out /= out.sum(axis=0) 
        return out
    
    def rasterize_seg_byorder(self, item, onehot=True):
        out = np.zeros((item['image_height'], item['image_width']), dtype=np.int)
        order = np.zeros((item['image_height'], item['image_width']), dtype=np.int)
        global_mask = mask_util.decode(item['mask']).astype(np.bool)
        order[~global_mask] = 255 # Background cannot be overridden!
        out[global_mask] = 1
        for i, s in enumerate(item['parts']):
            mask = mask_util.decode(s['mask']).astype(np.bool)
            part_order = vg3k_class_set.index(s['class'])+1
            replace_mask = mask & (order < part_order)
            out[replace_mask] = s['class_id']+2
            order[replace_mask] = part_order

        if onehot:
            out_oh = np.zeros((2 + self.num_parts, item['image_height'], item['image_width']), dtype=np.float32)
            for i in range(out_oh.shape[0]):
                out_oh[i, out == i] = 1
            return out_oh
        else:
            return out
    
    def forward_img(self, idx):
        if idx >= self.num_imgs:
            assert self.add_flipped
            assert idx < 2*self.num_imgs
            idx = idx - self.num_imgs
            force_flip = True
        else:
            force_flip = False
        item = self.detections[idx]
        
        img_path_rel = item['image_path']
        img_path = img_path_rel
        mask = mask_util.decode(item['mask'])
        bbox = item['bbox'].flatten()
        
        img = imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(mask, 2)
        
        if self.enable_seg:
            if self.rasterize_argmax:
                seg = self.rasterize_seg_byorder(item).transpose(1, 2, 0)
            else:
                seg = self.rasterize_seg(item).transpose(1, 2, 0)
        else:
            seg = np.zeros_like(img[:, :, :2]) # Dummy part segmentation (all background)
            seg[:, :, 0] = 1
        
        if self.gt_available and idx in self.gt:
            data_sfm = self.gt[idx]
            sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]
            sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
            sfm_rot[3, 3] = 1
            sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)
            z0 = np.zeros(1) # Dummy
            w = np.zeros(1) # Dummy
            semi_mask = np.zeros(1) # Dummy
        elif not self.unfiltered and self.poses is not None:
            # Rotation is already a quaternion, no need to further process it
            sfm_pose = [self.poses['s'][idx].numpy(), self.poses['t'][idx].numpy(), self.poses['R'][idx].numpy()]
            z0 = self.poses['z0'][idx].numpy()
            w = self.poses['w'][idx].numpy()
            semi_mask = self.semi_mask[idx].numpy()
        else:
            # Dummy pose (pose estimation mode)
            sfm_pose = [np.zeros(1), np.zeros(2), np.zeros(4)]
            sfm_pose[2][0] = -1000
            z0 = np.zeros(1)
            w = np.zeros(1)
            semi_mask = np.zeros(1)
            
        kp = np.zeros((1, 3))
        
        # Peturb bbox
        if self.is_train:
            jf = self.jitter_frac
        else:
            jf = 0
            
        bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=jf)
        bbox = image_utils.square_bbox(bbox)
        true_resolution = bbox[2] - bbox[0] + 1
        
        # crop image around bbox, translate kps
        vis = np.array([0], dtype=np.int)
        if self.poses is not None:
            # important! sfm_pose must not be overwritten -- it is already cropped
            img, mask, kp, _ = self.crop_image(img, mask, bbox, kp, vis, copy.deepcopy(sfm_pose))
        else:
            img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)
        
        seg_crop_bg = image_utils.crop(seg[:, :, :1], bbox, bgval=1)
        seg_crop_fg = image_utils.crop(seg[:, :, 1:], bbox, bgval=0)
        seg = np.concatenate((seg_crop_bg, seg_crop_fg), axis=2)
        
        mirrored = force_flip or (self.is_train and (torch.randint(0, 2, size=(1,)).item() == 1))

        # scale image, and mask. And scale kps.
        if self.poses is not None:
            # important! sfm_pose must not be overwritten -- it is already cropped
            sfm_pose_ref = copy.deepcopy(sfm_pose)
            img_ref, mask_ref, kp_ref, _ = self.scale_image(img.copy(), mask.copy(),
                                                           kp.copy(), vis.copy(),
                                                           copy.deepcopy(sfm_pose),
                                                           self.img_sizes[0])
        else:
            img_ref, mask_ref, kp_ref, sfm_pose_ref = self.scale_image(img.copy(), mask.copy(),
                                                                       kp.copy(), vis.copy(),
                                                                       copy.deepcopy(sfm_pose),
                                                                       self.img_sizes[0])
        
        scale = self.img_sizes[0] / float(max(seg.shape[0], seg.shape[1]))
        seg_scaled = []
        for i in range(seg.shape[-1]):
            seg_scale_tmp, _ = image_utils.resize_img(seg[:, :, i:i+1], scale)
            seg_scaled.append(seg_scale_tmp)
        seg = np.stack(seg_scaled, axis=2)
        
        if mirrored:
            if self.poses is not None:
                img_ref, mask_ref, kp_ref, _ = self.mirror_image(img_ref, mask_ref, kp_ref, copy.deepcopy(sfm_pose_ref))
                # Flip pose manually
                sfm_pose_ref[2] *= [1, 1, -1, -1]
                sfm_pose_ref[1] *= [-1, 1]
            else:
                img_ref, mask_ref, kp_ref, sfm_pose_ref = self.mirror_image(img_ref, mask_ref, kp_ref, sfm_pose_ref)
            seg = seg[:, ::-1, :].copy()

        # Normalize kp to be [-1, 1]
        img_h, img_w = img_ref.shape[:2]
        kp_norm, _ = self.normalize_kp(kp_ref, copy.deepcopy(sfm_pose_ref), img_h, img_w)

        # Finally transpose the image to 3xHxW
        img_ref = np.transpose(img_ref, (2, 0, 1))
        seg = np.transpose(seg, (2, 0, 1))
        
        # Compute other resolutions (if requested)
        extra_res = {}
        for res in self.img_sizes[1:]:
            if self.poses is not None:
                sfm_pose2 = copy.deepcopy(sfm_pose)
                img2, mask2, kp2, _ = self.scale_image(img.copy(), mask.copy(),
                                                       kp.copy(), vis.copy(),
                                                       copy.deepcopy(sfm_pose),
                                                       res)
            else:
                img2, mask2, kp2, sfm_pose2 = self.scale_image(img.copy(), mask.copy(),
                                                               kp.copy(), vis.copy(),
                                                               copy.deepcopy(sfm_pose),
                                                               res)
                
            if mirrored:
                if self.poses is not None:
                    img2, mask2, kp2, _ = self.mirror_image(img2, mask2, kp2, copy.deepcopy(sfm_pose2))
                    # Flip pose manually
                    sfm_pose2[2] *= [1, 1, -1, -1]
                    sfm_pose2[1] *= [-1, 1]
                else:
                    img2, mask2, kp2, sfm_pose2 = self.mirror_image(img2, mask2, kp2, sfm_pose2)
                
            img2 = np.transpose(img2, (2, 0, 1))
            extra_res[res] = (img2, mask2)

        return img_ref, kp_norm, mask_ref, sfm_pose_ref, mirrored, img_path_rel, seg, z0, w, semi_mask, extra_res
    
    def get_paths(self):
        paths = []
        for item in self.detections:
            paths.append(item['image_path'])
        if self.add_flipped:
            paths += paths
        return paths