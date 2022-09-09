# modified from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py

import os
import json
from turtle import down
import torch
import numpy as np
import imageio
from tqdm import tqdm
from skimage.transform import rescale

import sys
sys.path.append('./')
from utils.io_util import load_mask, load_rgb, glob_imgs, load_rgb_exr, load_mask_from_rembg,load_mask_cvat
from utils.rend_util import rot_to_quat, load_K_Rt_from_P

from src.preprocess_camera import get_normalization
from src.acquisition import imread_raw, preprocess_raw
from src.utils import linear_rgb_to_srgb, srgb_to_linear_rgb

import llff.poses.colmap_read_model as read_model

def string_line_to_vec(line):
    return np.array([float(item) 
                        for item in line.split()])

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 scale_radius=-1,
                 process_colmap=False,
                 normalize_cam=False,
                 processed=True,
                 save_masks=False,
                 save_horizon=False,
                 no_poses=False,
                 ):

        assert os.path.exists(data_dir), "Data directory is empty"

        self.instance_dir = data_dir
        self.train_cameras = train_cameras
        self.downscale = downscale
        self.no_poses = no_poses

        # Load images and mask
        image_dir = '{0}/images'.format(self.instance_dir)
        image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        self.has_mask = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
        mask_paths = sorted(glob_imgs(mask_dir))
        horizon_mask_dir = '{0}/horizon_mask'.format(self.instance_dir)
        horizon_mask_paths = sorted(glob_imgs(horizon_mask_dir))
        
        if not processed:
            raw_dir = '{0}/images_raw'.format(self.instance_dir)
            raw_paths = sorted(glob_imgs(raw_dir))
        
        if save_masks:
            os.makedirs('{0}/masks'.format(self.instance_dir), exist_ok=True)

        self.n_images = len(image_paths)

        # determine width, height
        self.downscale = downscale
        # _, self.H, self.W = tmp_rgb.shape
        raw_H, raw_W = (2048, 2448)
        if not processed:
            self.H, self.W = int(raw_H//(2*downscale)), int(raw_W//(2*downscale))
        else:
            self.H, self.W = int(raw_H//downscale), int(raw_W//downscale)
        tmp_rgb = load_rgb(image_paths[0], downscale) # Image used for Colmap
        _, rgb_H, rgb_W = tmp_rgb.shape
        scale_int = rgb_H/self.H # If intrisic matrix calibrated from Colmap has different dimension

        scene_name = self.instance_dir.split('/')[-2] # The raw data here starts with scene name
        self.scene_name = scene_name
        if not no_poses:
            self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
            if cam_file is not None:
                self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)
        
            if process_colmap:
                camera_dict, pts3d = self.poses_from_colmap()
                if normalize_cam:
                    # Requires masks
                    # get_normalization(self.instance_dir)
                    # Manually set using the mesh reconstructed for unnormalized case
                    # if scene_name.startswith("cricket_ball"):
                    if 1:
                        pts3d_center =  pts3d.mean(0)
                        # Manual translation
                        # pts3d_center[0] -= 0.1
                        # pts3d_center[2] -= 0.3
                        scale_mat = np.eye(4)
                        scale_mat[:3, 3] = pts3d_center
                        pts3d -= pts3d_center[None]
                    else:
                        raise Exception(f'Invalid scene_name {scene_name} for normalization')
                    for k in camera_dict.keys():
                        if k.startswith('scale_mat_'):
                            camera_dict[k] = scale_mat
                np.savez(self.cam_file, **camera_dict)
                np.savez(f'{self.instance_dir}/pts3d.npz', pts3d=pts3d)
            else:
                camera_dict = np.load(self.cam_file)
                pts3d = np.load(f'{self.instance_dir}/pts3d.npz')['pts3d']
            self.pts3d = pts3d
        
        self.image_paths = image_paths
        # Order of images based on ones that are calibrated
        if not no_poses:
            image_names = sorted([k[10:] for k in camera_dict.keys() if k.startswith('scale_mat_')])
        else:
            # Order of these mats is same as order of images in img_paths
            image_names = [image_path.split('/')[-1].split('.')[0] for image_path in self.image_paths]

        self.n_images = len(image_names)
        if not no_poses:
            scale_mats = [camera_dict[f'scale_mat_{image_name}'].astype(np.float32) for image_name in image_names]
            world_mats = [camera_dict[f'world_mat_{image_name}'].astype(np.float32) for image_name in image_names]

            self.intrinsics_all = []
            self.c2w_all = []
            cam_center_norms = []
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(P)
                cam_center_norms.append(np.linalg.norm(pose[:3,3]))
            
                # # Change intrinsics to match convention
                # intrinsics[0,0] = -intrinsics[0,0]
                # intrinsics[1,1] = -intrinsics[1,1]

                # downscale intrinsiccake
                intrinsics[0, 2] /= downscale*scale_int
                intrinsics[1, 2] /= downscale*scale_int
                intrinsics[0, 0] /= downscale*scale_int
                intrinsics[1, 1] /= downscale*scale_int
                # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale
            
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.c2w_all.append(torch.from_numpy(pose).float())
            max_cam_norm = max(cam_center_norms)
            if scale_radius > 0:
                for i in range(len(self.c2w_all)):
                    self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)


        self.s0_images = []
        self.s1_images = []
        self.s2_images = []
        self.rgb_images = []
        self.object_masks = []
        self.horizon_masks=[]
        if not processed:
            # NOTE: no trailing / might break this
            for image_name in tqdm(image_names, desc='loading raw...'):
                img_raw = imread_raw(f'{self.instance_dir}/images_raw/{int(image_name):02d}.raw', (raw_W, raw_H))
                out = preprocess_raw(img_raw, 0.5)
                rs_img = lambda x: rescale(x, 1./downscale, anti_aliasing=True, multichannel=True) \
                                    if downscale !=1 else x
                s0 = rs_img(out['stokes'][...,0]).reshape(-1,3)
                s1 = rs_img(out['stokes'][...,1]).reshape(-1,3)
                s2 = rs_img(out['stokes'][...,2]).reshape(-1,3)
                self.s0_images.append(torch.from_numpy(s0).float())
                self.s1_images.append(torch.from_numpy(s1).float())
                self.s2_images.append(torch.from_numpy(s2).float())
                self.rgb_images.append(torch.clamp(linear_rgb_to_srgb(0.5*torch.from_numpy(s0).float()),
                                                    min=0.0005, max=1.))

        else:
            for image_idx in tqdm(range(len(image_names)), desc='loading processed ...'):
                image_name = image_names[image_idx]
                path = image_paths[image_idx]
                rgb = load_rgb(path, downscale)
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                # self.rgb_images.append(torch.from_numpy(rgb).float())
                # self.rgb_images.append((torch.from_numpy(rgb).float()))

                s0 = 0.5*load_rgb(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0.hdr', downscale) 
                s0p1 = 0.5*load_rgb(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0p1.hdr', downscale)
                s0p2 = 0.5*load_rgb(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0p2.hdr', downscale)
                s1 = s0p1-s0
                s2 = s0p2-s0
                s0 = s0.reshape(3,-1).transpose(1,0)
                s1 = s1.reshape(3,-1).transpose(1,0)
                s2 = s2.reshape(3,-1).transpose(1,0)
                # self.rgb_images.append(torch.from_numpy(s0).float())
                if len(mask_paths)>0:
                    corrected_mask_path = f'{self.instance_dir}/mask_corrections/{int(image_name):02d}.png'
                    if os.path.exists(corrected_mask_path):
                        object_mask = load_mask_from_rembg(corrected_mask_path,downscale)
                    else:
                        object_mask = load_mask_from_rembg(f'{self.instance_dir}/mask/{int(image_name):02d}.png', 
                                                       downscale)
                    if save_masks:
                        imageio.imwrite(f'{self.instance_dir}/masks/{int(image_name):02d}.jpg',255*object_mask.astype('uint8'))
                    object_mask_res = object_mask.reshape(-1)
                    self.object_masks.append(torch.from_numpy(object_mask_res).to(dtype=torch.bool))
                    # Black bg
                    # self.s0_images.append(torch.from_numpy(object_mask[:,None]*s0).float())
                    # self.s1_images.append(torch.from_numpy(object_mask[:,None]*s1).float())
                    # self.s2_images.append(torch.from_numpy(object_mask[:,None]*s2).float())
                    # self.rgb_images.append(torch.clamp(linear_rgb_to_srgb(torch.from_numpy(object_mask[:,None]*s0).float()),
                    #                                     min=0.0, max=1.))
                    # White bg
                    # self.s0_images.append(torch.from_numpy(object_mask[:,None]*s0 + (1-object_mask)[:,None]).float())
                    # self.s1_images.append(torch.from_numpy(object_mask[:,None]*s1).float())
                    # self.s2_images.append(torch.from_numpy(object_mask[:,None]*s2).float())
                    # self.rgb_images.append(torch.clamp(linear_rgb_to_srgb(torch.from_numpy(object_mask[:,None]*s0+ (1-object_mask)[:,None]).float()),
                    #                                     min=0.0, max=1.))
                self.s0_images.append(torch.from_numpy(s0).float())
                self.s1_images.append(torch.from_numpy(s1).float())
                self.s2_images.append(torch.from_numpy(s2).float())
                # self.rgb_images.append(torch.clamp(linear_rgb_to_srgb(torch.from_numpy(s0).float()),
                #                                     min=0.0, max=1.))
                self.rgb_images.append(torch.from_numpy(s0).float())
                if len(horizon_mask_paths)>0:
                    horizon_mask = load_mask_cvat(f'{self.instance_dir}/horizon_mask/{int(image_name):02d}.png',
                                                        downscale)
                    if save_masks and save_horizon:
                        imageio.imwrite(f'{self.instance_dir}/masks/{int(image_name):02d}.jpg',((object_mask+horizon_mask)*255).astype('uint8'))
                    horizon_mask_res = horizon_mask.reshape(-1)
                    self.horizon_masks.append(torch.from_numpy(horizon_mask_res).to(dtype=torch.bool))

        self.processed = processed

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            # "object_mask": self.rgb_images[idx],# TODO: fix with computed masks or zeros
            "intrinsics": self.intrinsics_all[idx],
        }


        ground_truth = {}
        ground_truth['s0'] = self.s0_images[idx]
        ground_truth['s1'] = self.s1_images[idx]
        ground_truth['s2'] = self.s2_images[idx]
        ground_truth['rgb'] = self.rgb_images[idx]

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]

        if len(self.object_masks)>0:
            sample["object_mask"] = self.object_masks[idx]
        if len(self.horizon_masks)>0:
            sample["horizon_mask"] = self.horizon_masks[idx]
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_1']

    def get_gt_pose(self, scaled=True):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        # Order of these mats is same as order of images in img_paths
        # image_names = [image_path.split('/')[-1].split('.')[0] for image_path in self.image_paths]
        # Order of images based on ones that are calibrated
        image_names = sorted([k[10:] for k in camera_dict.keys() if k.startswith('scale_mat_')])
        scale_mats = [camera_dict[f'scale_mat_{image_name}'].astype(np.float32) for image_name in image_names]
        world_mats = [camera_dict[f'world_mat_{image_name}'].astype(np.float32) for image_name in image_names]
        # scale_mats = [camera_dict[f'scale_mat_{idx+1}'].astype(np.float32) for idx in range(self.n_images)]
        # world_mats = [camera_dict[f'world_mat_{idx+1}'].astype(np.float32) for idx in range(self.n_images)]

        c2w_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            c2w_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        # Order of these mats is same as order of images in img_paths
        # image_names = [image_path.split('/')[-1].split('.')[0] for image_path in self.image_paths]
        # Order of images based on ones that are calibrated
        image_names = sorted([k[10:] for k in camera_dict.keys() if k.startswith('scale_mat_')])
        scale_mats = [camera_dict[f'scale_mat_{image_name}'].astype(np.float32) for image_name in image_names]
        world_mats = [camera_dict[f'world_mat_{image_name}'].astype(np.float32) for image_name in image_names]
        # scale_mats = [camera_dict[f'scale_mat_{idx+1}'].astype(np.float32) for idx in range(self.n_images)]
        # world_mats = [camera_dict[f'world_mat_{idx+1}'].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

    def poses_from_colmap(self):
        # Adapted from llff pose_utils.py
        realdir = self.instance_dir
        camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
        camdata = read_model.read_cameras_binary(camerasfile)

        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]
        print( 'Cameras', len(cam))

        H, W = cam.height, cam.width
        f,x0,y0= cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f, 0., x0],
                      [0., f, y0],
                      [0., 0., 1.]])


        points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
        pts3d = read_model.read_points3d_binary(points3dfile)
        pts3d_xyz = np.array([pts3d[k].xyz for k in pts3d.keys()]) # N_pts x 3

        imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
        imdata = read_model.read_images_binary(imagesfile)
        names = [imdata[k].name for k in imdata]
        print( 'Images #', len(names))
        cam_dict = {}
        for k in imdata:
            im = imdata[k]
            image_name = im.name.split('.')[0]
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3,1])

            # Create transformation matrix
            trans_mtx = np.zeros((4,4))
            trans_mtx[3,3] = 1
            trans_mtx[:3,:3] = R
            trans_mtx[:3, [3]] = t

            P = K @ trans_mtx[:3] # 3x4
            
            # [0 0 0 1] concatenated
            P = np.concatenate([P,
                np.array([[0.,0.,0.,1]])],0) # 4x4 

            # Convert conventions
            # Mitsuba X1 Up: +y Towards camera: +z Right handed
            # DTU dataset Y1: Up: -z Towards camera: +x Right handed
            # Conversion Matrix
            C = np.array([[0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [1., 0., 0.,  0.],
                          [0., 0., 0.,  1.]])

            # R = np.array([[0., 1., 0., 0.],
            #               [1., 0., 0., 0.],
            #               [0., 0., -1.,  0.],
            #               [0., 0., 0.,  1.]])
            P = P@C

            cam_dict[f'world_mat_{image_name}'] = P
            cam_dict[f'scale_mat_{image_name}'] = np.eye(4)
        # Coordinate conversion
        pts3d_xyz = pts3d_xyz@C[:3,:3]
        return cam_dict, pts3d_xyz

if __name__ == "__main__":
    dataset = SceneDataset(False, 'data/our_dataset/a2_ceramic_owl_v2/',
                           process_colmap=True, normalize_cam=True,
                           processed=True, downscale=1,
                           scale_radius=3., save_masks=True,
                           no_poses=True)
    # Processing for NeuralPIL
    # dataset = SceneDataset(False, 'data/our_dataset/black_vase_v1/',
    #                        process_colmap=False, normalize_cam=True,
    #                        processed=True, downscale=1,
    #                        scale_radius=3., save_masks=True, save_horizon=True)
    if not dataset.no_poses:
        c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
        extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
        camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
        # Save points for interactive visualizaiton
        from scipy.io import savemat
        savemat(f'viz/our_{dataset.scene_name}.mat',{'c2w':c2w, 'pts3d':dataset.pts3d})
        from tools.vis_camera import visualize
        visualize(camera_matrix, extrinsics, 'ours', pts3d=dataset.pts3d)
    H, W = dataset.H, dataset.W
    def to_img_stack_exr(x):
        return (x.reshape(-1, H, W,3).cpu().numpy()).astype('float32')
    def to_img_stack(x):
        return (x.reshape(-1, H, W,3).cpu().numpy()*255.).astype('uint8')
    def to_mask_stack(x):
        return x.reshape(-1, H, W).cpu().numpy() + 0.
    target_rgb_stack = to_img_stack(torch.stack(dataset.rgb_images,0))
    if len(dataset.object_masks)>0:
        target_mask_stack = to_mask_stack(torch.stack(dataset.object_masks,0))
    os.makedirs(f'viz/our_{dataset.scene_name}_rgb',exist_ok=True)
    import imageio
    for idx in range(target_rgb_stack.shape[0]): 
    # for idx in [0,1]: 
        imageio.imwrite(f'viz/our_{dataset.scene_name}_rgb/rgb_{idx:03d}.png',target_rgb_stack[idx])
        # imageio.imwrite(f'viz/our_{dataset.scene_name}_rgb/mask_{idx:03d}.png',target_mask_stack[idx])
    imageio.mimwrite(f'viz/our_{dataset.scene_name}_rgb.mp4', target_rgb_stack,fps=5, quality=8)
    from src.polarization import cues_from_stokes, colorize_cues
    # Convert to polarimetric cues
    #  Same as in train.py
    target_stokes = torch.stack([torch.stack(dataset.s0_images,0),
                                 torch.stack(dataset.s1_images,0), 
                                 torch.stack(dataset.s2_images,0)],
                                 -1)
    target_cues = colorize_cues(cues_from_stokes(target_stokes),
                                gamma_s0=True)
    os.makedirs(f'viz/our_{dataset.scene_name}_cues/',exist_ok=True)
    # Also plot net polarized intensity
    for idx in range(target_stokes.shape[0]):
        pol_int = linear_rgb_to_srgb(torch.clip(torch.hypot(target_stokes[idx,...,1],
                                                            target_stokes[idx,...,2]),
                                     0,1))
        unpol_int = linear_rgb_to_srgb(torch.clip(target_stokes[idx, ...,0] - 
                                                  torch.hypot(target_stokes[idx,...,1],
                                                              target_stokes[idx,...,2]),
                                     0,1))
        pol_int_exr = (torch.clip(torch.hypot(target_stokes[idx,...,1],
                                              target_stokes[idx,...,2]),
                                     0,1))
        unpol_int_exr = (torch.clip(target_stokes[idx,...,0] -
                                    torch.hypot(target_stokes[idx,...,1],
                                                target_stokes[idx,...,2]),
                                     0,1))
        imageio.imwrite(f'viz/our_{dataset.scene_name}_cues/pol_int_{idx:03d}.png',
                        to_img_stack(pol_int)[0])
        imageio.imwrite(f'viz/our_{dataset.scene_name}_cues/pol_int_{idx:03d}.exr',
                        to_img_stack_exr(pol_int_exr)[0])
        imageio.imwrite(f'viz/our_{dataset.scene_name}_cues/unpol_int_{idx:03d}.png',
                        to_img_stack(unpol_int)[0])
        imageio.imwrite(f'viz/our_{dataset.scene_name}_cues/unpol_int_{idx:03d}.exr',
                        to_img_stack_exr(unpol_int_exr)[0])
    for cue_name, cue_val in target_cues.items():
        cue_val_stack = to_img_stack(cue_val)
        imageio.mimwrite(f'viz/our_{dataset.scene_name}_cues_{cue_name}.mp4',cue_val_stack,fps=5,quality=8)
        for idx in range(cue_val.shape[0]):
            imageio.imwrite(f'viz/our_{dataset.scene_name}_cues/{cue_name}_{idx:03d}.png',cue_val_stack[idx])
    import pdb; pdb.set_trace()