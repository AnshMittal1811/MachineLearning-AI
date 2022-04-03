import torch
import skimage
import skimage.filters
import skimage.transform
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import numpy as np
import os
from tqdm.autonotebook import tqdm
import json
import re
import coord_transforms
import rot_utils


class Implicit1DWrapper(torch.utils.data.Dataset):
    def __init__(self, range, fn, grad_fn=None, integral_fn=None, sampling_density=100,
                 train_every=10, jitter=False):

        avg = (range[0] + range[1]) / 2
        coords = self.get_samples(range, sampling_density)
        self.fn_vals = fn(coords)
        self.train_idx = torch.arange(0, coords.shape[0], train_every).float()

        self.grid = coords
        self.grid.requires_grad_(True)
        self.jitter = jitter
        self.range = range

        if grad_fn is None:
            grid_gt_with_grad = coords
            grid_gt_with_grad.requires_grad_(True)
            fn_vals_with_grad = fn((grid_gt_with_grad * (range[1] - avg)) + avg)
            gt_gradient = torch.autograd.grad(fn_vals_with_grad, [grid_gt_with_grad],
                                              grad_outputs=torch.ones_like(grid_gt_with_grad), create_graph=True,
                                              retain_graph=True)[0]
        else:
            gt_gradient = grad_fn(coords)

        self.integral_fn = integral_fn
        self.integral_vals = None
        if integral_fn:
            self.integral_vals = integral_fn(coords)

        self.gt_gradient = gt_gradient.detach()

    def get_samples(self, range, sampling_density):
        num = int(range[1] - range[0])*sampling_density
        coords = np.linspace(start=range[0], stop=range[1], num=num)
        coords.astype(np.float32)
        coords = torch.Tensor(coords).view(-1, 1)
        return coords

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        if self.jitter:
            jitter = torch.rand_like(self.grid) * (self.grid[1] - self.grid[0])
        else:
            jitter = 0.

        return {'idx': self.train_idx, 'coords': self.grid + jitter}, \
               {'func': self.fn_vals, 'gradients': self.gt_gradient,
                'coords': self.grid, 'integral': self.integral_vals}


def polynomial_1(coords):
    return .1*coords**5 - .2*coords**4 + .2*coords**3 - .4*coords**2 + .1*coords


def polynomial_1_integral(coords):
    return .1/6*coords**6 - .2/5*coords**5 + .2/4*coords**4 - .4/3*coords**3 + .1/2*coords**2


def sinc(coords):
    coords[coords == 0] += 1
    return torch.div(torch.sin(20*coords), 20*coords)


def linear(coords):
    return 1.0 * coords


def xcosx(coords):
    return coords * torch.cos(coords)


def int_xcosx(coords):
    return coords*torch.sin(coords) + torch.cos(coords)


class SheppLoganPhantomRadonTransformed(Dataset):
    def __init__(self, rho_resolution, theta_resolution=180):
        super().__init__()
        self.img = Image.fromarray(skimage.data.shepp_logan_phantom())
        self.img_channels = 1

        if(rho_resolution != self.img.size[0]):
            size = (int(rho_resolution),) * 2
            self.img = self.img.resize(size, resample=Image.BILINEAR)

        self.img = np.array(self.img)
        self.rho_res = min(self.img.shape)
        self.theta_res = theta_resolution
        theta = np.linspace(0, 180, self.theta_res)
        self.radon = skimage.transform.radon(self.img, theta, circle=True) * 2/self.rho_res
        self.iradon = skimage.transform.iradon(self.radon, theta=theta, circle=True) * self.rho_res/2

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.radon, self.iradon, self.img


class Implicit2DRadonTomoWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, mc_resolution,
                 rho_resolution, theta_resolution, subsampling_factor=4):
        self.dataset = dataset
        self.mc_resolution = mc_resolution
        self.rho_res = rho_resolution
        self.theta_res = theta_resolution

        self.subsampling_factor = subsampling_factor
        if self.subsampling_factor > 1:
            self.is_subsampled = True
        else:
            self.is_subsampled = True

        self.rho_res_subsampled = self.rho_res
        self.theta_res_subsampled = self.theta_res//self.subsampling_factor

        # sampling without replacement ensures all the (rho,theta) are sampled once
        self.idx = 0
        self._generate_rho_theta_coords()
        self._generate_rho_theta_coords_subsampled()

        # compute start and stop sampling points
        self.lin_rho = torch.linspace(-1, 1, self.rho_res_subsampled)
        self.lin_theta = torch.linspace(0, np.pi, self.theta_res_subsampled)

        self.lin_rho_no_sub = torch.linspace(-1, 1, self.rho_res)
        self.lin_theta_no_sub = torch.linspace(0, np.pi, self.theta_res)

        rho, theta = torch.meshgrid(self.lin_rho, self.lin_theta)
        self.vals = torch.stack([rho.reshape(-1), theta.reshape(-1)], 0)

        # For a line parameterized by (rho,theta=0) the 2 intersection points
        # beg_pts and end_pts with the circle centered at the origin and of radius 1
        # are (rho,sqrt(1-rho^2)) and (rho,-sqrt(1-rho^2))
        self.beg_pts = torch.stack([rho, -torch.sqrt(1.-rho**2)])  # X,Y coordinates
        self.end_pts = torch.stack([rho, torch.sqrt(1.-rho**2)])  # X,Y coordinates

        # Then to get the coordinates of the two intersection of the line parameterized bu
        # (rho,theta) and the circle, we simply rotate the previous frame by theta
        # -- rotate beg pts
        x = self.beg_pts[0, :, :].clone()
        y = self.beg_pts[1, :, :].clone()
        self.beg_pts[0, :, :] = torch.cos(theta) * x - torch.sin(theta) * y  # rot x
        self.beg_pts[1, :, :] = torch.sin(theta) * x + torch.cos(theta) * y  # rot y
        # -- rotate end pts
        x = self.end_pts[0, :, :].clone()
        y = self.end_pts[1, :, :].clone()
        self.end_pts[0, :, :] = torch.cos(theta) * x - torch.sin(theta) * y  # rot x
        self.end_pts[1, :, :] = torch.sin(theta) * x + torch.cos(theta) * y  # rot y

    def __len__(self):
        if(self.is_subsampled):
            return self.rho_res_subsampled*self.theta_res_subsampled  # rho res. x theta res.
        else:
            return self.rho_res*self.theta_res  # rho res. x theta res.

    def _generate_rho_theta_coords(self):
        lin_rho = torch.arange(0, self.rho_res, 1)
        lin_theta = torch.arange(0, self.theta_res, 1)
        rho, theta = torch.meshgrid(lin_rho, lin_theta)

        self.coords = torch.stack([rho.reshape(-1).long(),
                                   theta.reshape(-1).long()],
                                  0)

    def _generate_rho_theta_coords_subsampled(self):
        lin_rho = torch.arange(0, self.rho_res_subsampled, 1)
        lin_theta = torch.arange(0, self.theta_res_subsampled, 1)
        rho, theta = torch.meshgrid(lin_rho, lin_theta)

        self.coords_subsampled = torch.stack([rho.reshape(-1).long(),
                                              theta.reshape(-1).long()], 0)

    def __getitem__(self, idx):
        radon_img, iradon_img, img = self.dataset[0]           # a single image
        radon_t_img = torch.from_numpy(radon_img)  # convert to tensor
        iradon_t_img = torch.from_numpy(iradon_img)

        # Samples on the line between p_beg and p_end
        # Use idx to sample without replacement
        if(self.is_subsampled):
            coords = self.coords_subsampled
        else:
            coords = self.coords
        rho_s, theta_s = coords[:, idx]

        t_samples = torch.rand([self.mc_resolution, 1])

        # rho, theta, t
        rho_samples = self.vals[0, idx].repeat(self.mc_resolution, 1)
        theta_samples = self.vals[1, idx].repeat(self.mc_resolution, 1)

        min_t = -torch.sqrt(1. - rho_samples[0]**2)
        max_t = torch.sqrt(1. - rho_samples[0]**2)

        t_vals = torch.linspace(0.0, 1.0, self.mc_resolution)[:, None]
        t_vals = min_t * (1.0 - t_vals) + max_t * t_vals

        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat((mids, t_vals[..., -1:]), dim=-1)
        lower = torch.cat((t_vals[..., :1], mids), dim=-1)

        # Stratified samples in those intervals.
        t_rand = torch.rand(t_vals.shape)
        t_samples = lower + (upper - lower) * t_rand
        ray_len = max_t - min_t

        # Ground truth
        radon_t_s = radon_t_img[rho_s, theta_s*self.subsampling_factor]

        in_dict = {'rho': rho_samples,
                   'theta': theta_samples, 't': t_samples}
        gt_dict = {'radon_integral': radon_t_s,
                   'radon_img': radon_t_img, 'iradon_img': iradon_t_img,
                   'img': img,
                   'theta': self.lin_theta_no_sub, 'rho': self.lin_rho_no_sub,
                   'rho_res': self.rho_res, 'ray_len': ray_len}
        return in_dict, gt_dict


# adapted from NeRF code https://github.com/bmild/nerf/blob/20a91e764a28816ee2234fcadb73bd59a613a44c/load_llff.py
class LLFFDataset(Dataset):
    def __init__(self, basedir="../data/nerf_llff_data/fern", mode='train',
                 final_render=False, scale_factor=8, bound_scale=0.75, recenter=True):
        self.mode = mode
        self.basedir = basedir
        self.final_render = final_render

        print("Starting loading images")
        poses, c2w, bounds, images = self.load_data(basedir, scale_factor, bound_scale, recenter)

        if self.final_render:
            poses = self.get_render_poses(poses, c2w, bounds)

        # transform to tensor
        transform_list = [torchvision.transforms.ToTensor()]
        transforms = torchvision.transforms.Compose(transform_list)
        images = torch.stack([transforms(img) for img in images], 0)
        images = images.permute((0, 2, 3, 1))
        images = torch.cat([images, torch.ones_like(images[..., 0:1])], dim=-1)  # add dummy alpha channels

        self.imgs = images
        self.img_shape = self.imgs[0].shape

        # set camera parameters
        H, W, focal = poses[0, :3, -1]
        self.poses = torch.from_numpy(poses[:, :3, :4])
        self.poses = torch.cat((self.poses,
                               torch.zeros(self.poses.shape[0], 1, self.poses.shape[2])),
                               dim=1)
        self.poses[:, -1, -1] = 1.

        camera_angle_x = np.arctan(2.*focal/W)*2.
        self.camera_params = {'H': H, 'W': W,
                              'camera_angle_x': camera_angle_x,
                              'focal': focal,
                              'near': 0.,
                              'far': 1.}

        # generate test set consistent with nerf paper
        idx_test = np.arange(self.imgs.shape[0])[::8]
        idx_train = np.array([i for i in np.arange(self.imgs.shape[0]) if i not in idx_test])

        if self.final_render:
            self.imgs = torch.ones((len(self.poses),
                                    self.imgs[0].shape[0],
                                    self.imgs[0].shape[1], 3))
        elif mode == 'train':
            self.imgs = self.imgs[idx_train]
            self.poses = self.poses[idx_train]
        else:
            self.imgs = self.imgs[idx_test]
            self.poses = self.poses[idx_test]

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def viewmatrix(self, z, up, pos):
        vec2 = self.normalize(z)
        vec1_avg = up
        vec0 = self.normalize(np.cross(vec1_avg, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def get_render_poses(self, poses, c2w, bounds):
        # get spiral
        up = self.normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bounds.min()*.9, bounds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2

        return self.render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    def render_path_spiral(self, c2w, up, rads, focal, zdelta, zrate, rots, N):
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w[:, 4:5]

        for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
            c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta),
                       -np.sin(theta*zrate), 1.]) * rads)
            z = self.normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([self.viewmatrix(z, up, c), hwf], 1))

        render_poses = np.array(render_poses).astype(np.float32)
        return render_poses

    def load_data(self, basedir, factor=8, bound_scale=0.75, recenter=True):
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bounds = poses_arr[:, -2:].transpose([1, 0])

        imgdir = os.path.join(basedir, f'images_{factor}')
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        assert poses.shape[-1] == len(imgfiles), 'Mismatch between poses and images'

        sh = np.array(Image.open(imgfiles[0])).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./factor

        imgs = [np.array(Image.open(f))[..., :3]/255. for f in imgfiles]
        imgs = np.stack(imgs, 0)

        # correct rotation matrix ordering
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = imgs
        bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

        # Rescale if bound_scale is provided
        sc = 1. if bound_scale is None else 1./(bounds.min() * bound_scale)
        poses[:, :3, 3] *= sc
        bounds *= sc

        def poses_avg(poses):
            hwf = poses[0, :3, -1:]
            center = poses[:, :3, 3].mean(0)
            vec2 = self.normalize(poses[:, :3, 2].sum(0))
            up = poses[:, :3, 1].sum(0)
            c2w = np.concatenate([self.viewmatrix(vec2, up, center), hwf], 1)
            return c2w

        if recenter:
            poses_ = poses+0
            bottom = np.reshape([0, 0, 0, 1.], [1, 4])
            c2w = poses_avg(poses)
            c2w = np.concatenate([c2w[:3, :4], bottom], -2)
            bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
            poses = np.concatenate([poses[:, :3, :4], bottom], -2)

            poses = np.linalg.inv(c2w) @ poses
            poses_[:, :3, :4] = poses[:, :3, :4]
            poses = poses_
            c2w = poses_avg(poses)

        return poses, c2w, bounds, images

    def set_mode(self, mode):
        self.mode = mode

    def get_img_shape(self):
        return self.img_shape

    def get_camera_params(self):
        return self.camera_params

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return {'img': self.imgs[item, ...],
                'pose': self.poses[item, ...]}


class DeepVoxelDataset(Dataset):
    def __init__(self, basedir="../data/deepvoxel/globe", mode='train',
                 idcs=[], resize_to=512, final_render=False):
        self.mode = mode
        self.basedir = basedir
        self.final_render = final_render
        self.resize_to = resize_to

        # Create splits
        if mode == 'train':
            all_poses = self.dir2poses(os.path.join(basedir, 'pose'))

            num_poses = all_poses.shape[0]
            all_idcs = np.random.permutation(num_poses)

            self.idcs = all_idcs[:num_poses//2]
            self.val_idcs = all_idcs[num_poses//2:]
            self.poses = all_poses[self.idcs, ...]

        elif mode == 'val':
            if idcs == []:
                print(f"You must provide the validation indices")
            all_poses = self.dir2poses(os.path.join(basedir, 'pose'))
            self.idcs = idcs
            self.poses = all_poses[self.idcs, ...]

        elif mode == 'test':
            all_poses = self.dir2poses(os.path.join(basedir, 'pose'))
            self.idcs = np.arange(0, all_poses.shape[0])
            self.poses = all_poses[self.idcs, ...]

        H = resize_to[0]
        W = resize_to[1]
        full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses \
            = self.parse_intrinsics(os.path.join(self.basedir, 'intrinsics.txt'), H)

        focal = full_intrinsic[0, 0]

        if mode != 'test':
            print("Starting loading images")
            transform_list = [torchvision.transforms.ToTensor()]
            if self.resize_to is not None:
                transform_list.insert(0, torchvision.transforms.Resize(resize_to,
                                                                       interpolation=Image.BILINEAR))
            transforms = torchvision.transforms.Compose(transform_list)

            imgfiles = [f for f in sorted(os.listdir(os.path.join(self.basedir, 'rgb'))) if f.endswith('png')]
            self.imgs = []
            for file in tqdm(imgfiles):
                img = Image.open(os.path.join(self.basedir, 'rgb', file))
                # we assume landscape format
                W, H = img.size
                top = 0
                bottom = H
                left = W//2 - H//2
                right = W//2 + H//2
                img = img.crop((left, top, right, bottom))  # crop center 1080x1080
                img_t = transforms(img)
                self.imgs.append(img_t.permute(1, 2, 0))
            print("Done loading images")

            self.img_shape = self.imgs[0].shape
            self.imgs = torch.stack(self.imgs, 0)[self.idcs, ...]
            self.imgs = torch.cat([self.imgs, torch.
                                  torch.ones_like(self.imgs[..., 0:1])],
                                  dim=-1)  # add dummy alpha channels

        self.poses = torch.from_numpy(self.poses)

        camera_angle_x = np.arctan(2.*focal/W)*2.
        self.camera_params = {'H': H, 'W': W,
                              'camera_angle_x': camera_angle_x,
                              'focal': focal,
                              'near': near_plane-1,
                              'far': near_plane+1}

    def parse_intrinsics(self, filepath, trgt_sidelength, invert_y=False):
        # Get camera intrinsics
        with open(filepath, 'r') as file:
            f, cx, cy = list(map(float, file.readline().split()))[:3]
            grid_barycenter = np.array(list(map(float, file.readline().split())))
            near_plane = float(file.readline())
            scale = float(file.readline())
            height, width = map(float, file.readline().split())

            try:
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        if world2cam_poses is None:
            world2cam_poses = False

        world2cam_poses = bool(world2cam_poses)
        cx = trgt_sidelength/2.
        cy = trgt_sidelength/2.
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f

        # Build the intrinsic matrices
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])

        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses

    def dir2poses(self, posedir):
        poses = np.stack(
            [self.load_pose(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0)
        transf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.],
        ])
        poses = poses @ transf
        poses = poses[:, :4, :4].astype(np.float32)
        return poses

    def load_pose(self, filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    def set_mode(self, mode):
        self.mode = mode

    def get_img_shape(self):
        if self.mode != 'test':
            return self.img_shape
        else:
            dummy = torch.ones((self.resize_to[0], self.resize_to[1], 3))
            return dummy.shape

    def get_camera_params(self):
        return self.camera_params

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, item):
        if self.mode == 'test':
            dummy = torch.ones((self.resize_to[0], self.resize_to[1], 3))
            return {'img': dummy,
                    'pose': self.poses[item, ...]}
        else:
            return {'img': self.imgs[item, ...],
                    'pose': self.poses[item, ...]}


class NerfBlenderDataset(Dataset):
    def __init__(self, basedir, mode='train',
                 splits=['train', 'val', 'test'],
                 select_idx=None,
                 testskip=1, resize_to=None, final_render=False,
                 ref_rot=None, d_rot=0, bounds=((-2, 2), (-2, 2), (0, 2))):
        self.mode = mode
        self.basedir = basedir
        self.resize_to = resize_to
        self.final_render = final_render
        self.bounds = bounds

        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        # Eventually transform the inputs
        transform_list = [torchvision.transforms.ToTensor()]
        if resize_to is not None:
            transform_list.insert(0, torchvision.transforms.Resize(resize_to,
                                                                   interpolation=Image.BILINEAR))
        transforms = torchvision.transforms.Compose(transform_list)

        # Gather images and poses
        self.all_imgs = {}
        self.all_poses = {}
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []

            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                if select_idx is not None:
                    if re.search('[0-9]+', frame['file_path']).group(0) != select_idx:
                        continue

                fname = os.path.join(basedir, frame['file_path'] + '.png')
                img = Image.open(fname)
                pose = torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32))

                if ref_rot is None:
                    img_t = transforms(img)
                    imgs.append(img_t.permute(1, 2, 0))
                    poses.append(pose)
                else:
                    dist_angle = rot_utils.dist_between_rotmats(ref_rot, pose[:3, :3].permute(1, 0))
                    if(dist_angle <= d_rot):
                        img_t = transforms(img)
                        imgs.append(img_t.permute(1, 2, 0))
                        poses.append(pose)

            self.all_imgs.update({s: imgs})
            self.all_poses.update({s: poses})

        if self.final_render:
            self.poses = [torch.from_numpy(self.pose_spherical(angle, -30.0, 4.0)).float()
                          for angle in np.linspace(-180, 180, 40 + 1)[:-1]]

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.camera_params = {'H': H, 'W': W,
                              'camera_angle_x': camera_angle_x,
                              'focal': focal,
                              'near': 2.0,
                              'far': 6.0}
        self.img_shape = imgs[0].shape

    # adapted from https://github.com/krrish94/nerf-pytorch
    # derived from original nerf repo (MIT License)
    def translate_by_t_along_z(self, t):
        tform = np.eye(4).astype(np.float32)
        tform[2][3] = t
        return tform

    def rotate_by_phi_along_x(self, phi):
        tform = np.eye(4).astype(np.float32)
        tform[1, 1] = tform[2, 2] = np.cos(phi)
        tform[1, 2] = -np.sin(phi)
        tform[2, 1] = -tform[1, 2]
        return tform

    def rotate_by_theta_along_y(self, theta):
        tform = np.eye(4).astype(np.float32)
        tform[0, 0] = tform[2, 2] = np.cos(theta)
        tform[0, 2] = -np.sin(theta)
        tform[2, 0] = -tform[0, 2]
        return tform

    def pose_spherical(self, theta, phi, radius):
        c2w = self.translate_by_t_along_z(radius)
        c2w = self.rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
        c2w = self.rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w

    def set_mode(self, mode):
        self.mode = mode

    def get_img_shape(self):
        return self.img_shape

    def get_camera_params(self):
        return self.camera_params

    def __len__(self):
        if self.final_render:
            return len(self.poses)
        else:
            return len(self.all_imgs[self.mode])

    def __getitem__(self, item):
        # render out trajectory (no GT images)
        if self.final_render:
            return {'img': torch.zeros(4),  # we have to pass something...
                    'pose': self.poses[item]}

        # otherwise, return GT images and pose
        else:
            return {'img': self.all_imgs[self.mode][item],
                    'pose': self.all_poses[self.mode][item]}


class Implicit6DMultiviewDataWrapper(Dataset):
    def __init__(self, dataset, img_shape, camera_params,
                 use_ndc=False, use_near_far_shift=True,
                 samples_per_ray=128,
                 samples_per_view=32000,
                 sobol_ray_sampling=False,
                 num_workers=4):

        self.dataset = dataset
        self.num_workers = num_workers

        self.img_shape = img_shape
        self.camera_params = camera_params

        self.use_ndc = use_ndc
        self.use_near_far_shift = use_near_far_shift

        self.samples_per_view = samples_per_view
        self.default_samples_per_view = samples_per_view
        self.samples_per_ray = samples_per_ray

        self._generate_rays_normalized()
        self._precompute_rays()
        self.sobol_ray_sampling = sobol_ray_sampling

        self.is_logging = False

        self.soboleng = [torch.quasirandom.SobolEngine(dimension=1, scramble=True,
                                                       seed=(torch.rand(1)*1000).long().item())
                         for i in range(num_workers)]

    def draw_sobol_samples(self, worker_id, N, len):
        return (self.soboleng[worker_id].draw(N) * len).long().squeeze()

    def toggle_logging_sampling(self):
        if self.is_logging:
            self.samples_per_view = self.default_samples_per_view
            self.is_logging = False
        else:
            self.samples_per_view = self.img_shape[0] * self.img_shape[1]
            self.is_logging = True

    def _generate_rays_normalized(self):
        rows = torch.arange(0, self.img_shape[0], dtype=torch.float32)
        cols = torch.arange(0, self.img_shape[1], dtype=torch.float32)
        g_rows, g_cols = torch.meshgrid(rows, cols)

        W = self.camera_params['W']
        H = self.camera_params['H']
        f = self.camera_params['focal']

        self.norm_rays = torch.stack([(g_cols-.5*W)/f,
                                      -(g_rows-.5*H)/f,
                                      -torch.ones_like(g_rows)],
                                     dim=2).view(-1, 3).permute(1, 0)

        self.num_rays_per_view = self.norm_rays.shape[1]

    def _precompute_rays(self):
        img_list = []
        pose_list = []

        ray_orgs_list = []
        ray_dirs_list = []
        ray_up_list = []

        ray_norm_dirs_list = []
        ray_norm_ups_list = []

        ray_flatrotmats_list = []
        pose_flatrotmats_list = []

        print('Precomputing rays...')
        for img_pose in tqdm(self.dataset):
            img = img_pose['img']
            img_list.append(img)

            pose = img_pose['pose']
            pose_list.append(pose)

            ray_dirs = pose[:3, :3].matmul(self.norm_rays).permute(1, 0)
            ray_dirs_list.append(ray_dirs)

            ray_up = pose[:3, :3].matmul(torch.tensor([[0.], [1.], [0.]]))
            ray_up_list.append(ray_up)

            ray_orgs = pose[:3, 3].repeat((self.num_rays_per_view, 1))
            ray_orgs_list.append(ray_orgs)

            ray_norm_dirs = ray_dirs / torch.sqrt(torch.sum(ray_dirs**2, dim=-1, keepdim=True))
            ray_norm_dirs_list.append(ray_norm_dirs)
            ray_up = ray_up.permute(1, 0)
            ray_norm_up = ray_up / torch.sqrt(torch.sum(ray_up ** 2, dim=-1, keepdim=True))
            ray_norm_ups = ray_norm_up.repeat(self.num_rays_per_view, 1)  # [num_rays,3]
            ray_norm_ups_list.append(ray_norm_ups)

            ray_rotmats = rot_utils.dirup_to_rotmat(ray_norm_dirs, ray_norm_ups).permute(0, 2, 1)
            ray_flatrotmats_list.append(ray_rotmats.reshape(-1, 9))

            pose_rotmat = pose[0:3, 0:3][None, :, :]
            pose_flatrotmats_list.append(pose_rotmat.reshape(-1, 9))

        self.all_imgs = torch.stack(img_list, dim=0)
        self.all_poses = torch.stack(pose_list, dim=0)

        self.all_ray_orgs = torch.stack(ray_orgs_list, dim=0)
        self.all_ray_dirs = torch.stack(ray_dirs_list, dim=0)
        self.all_up_dirs = torch.stack(ray_up_list, dim=0).permute(0, 2, 1)
        self.all_ray_norm_dirs = torch.stack(ray_norm_dirs_list, dim=0)
        self.all_up_norm_dirs = torch.stack(ray_norm_ups_list, dim=0)

        self.all_ray_flatrotmats = torch.stack(ray_flatrotmats_list, dim=0)
        self.all_pose_flatromats = torch.stack(pose_flatrotmats_list, dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_i = self.all_imgs[idx, ...]
        pose_i = self.all_poses[idx, ...]
        ray_dirs_i = self.all_ray_dirs[idx, ...]
        ray_orgs_i = self.all_ray_orgs[idx, ...]
        ray_flatrotmats_i = self.all_ray_flatrotmats[idx, ...]
        view_samples_i = img_i.reshape(-1, 4)

        # Eventually subsample the pixels
        if not self.is_logging:
            num_rays_to_sample = self.samples_per_view

            # First sample randomly
            if self.sobol_ray_sampling:
                # use sobol sampling instead
                worker_id = torch.utils.data.get_worker_info().id
                pix_samples = self.draw_sobol_samples(worker_id, num_rays_to_sample, ray_orgs_i.shape[0])
            else:
                pix_samples = torch.randperm(ray_orgs_i.shape[0])[:num_rays_to_sample]

            view_samples = view_samples_i[pix_samples, ...]
            ray_orgs = ray_orgs_i[pix_samples, ...]
            ray_dirs = ray_dirs_i[pix_samples, ...]
            ray_flatrotmats = ray_flatrotmats_i[pix_samples, ...]

        else:
            view_samples = view_samples_i
            ray_orgs = ray_orgs_i
            ray_dirs = ray_dirs_i
            ray_flatrotmats = ray_flatrotmats_i

        # Transform coordinate systems
        camera_params = self.dataset.get_camera_params()

        if self.use_ndc:
            transform = coord_transforms.ToNormalizedDeviceCoordinates(camera_params)
            ray_orgs, ray_dirs = transform(ray_orgs, ray_dirs)

        ray_dirs = ray_dirs[:, None, :]
        ray_orgs = ray_orgs[:, None, :]

        t_vals = torch.linspace(0.0, 1.0, self.samples_per_ray)
        t_vals = camera_params['near'] * (1.0 - t_vals) + camera_params['far'] * t_vals
        t_vals = t_vals[None, :].repeat(self.samples_per_view, 1)

        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat((mids, t_vals[..., -1:]), dim=-1)
        lower = torch.cat((t_vals[..., :1], mids), dim=-1)

        # Stratified samples in those intervals.
        t_rand = torch.rand(t_vals.shape)
        t_vals = lower + (upper - lower) * t_rand
        t = t_vals[..., None]

        ray_samples = ray_orgs + ray_dirs * t_vals[..., None]

        # Compute distance samples from orgs
        dist_samples_to_org = torch.sqrt(torch.sum((ray_samples-ray_orgs)**2, dim=-1, keepdim=True))

        # broadcast tensors
        ray_flatrotmats = ray_flatrotmats[:, None, :]

        in_dict = {'ray_origins': ray_orgs,
                   'ray_orientations': ray_flatrotmats[..., 0:6],
                   'ray_directions': ray_dirs,
                   'ray_directions_norm': ray_dirs,
                   't': t}
        meta_dict = {'zs': dist_samples_to_org}
        gt_dict = {'pixel_samples': view_samples}

        if not self.is_logging:
            gt_dict.update({'view_idx': idx*torch.ones_like(pix_samples)[:, None],
                            'pixel_idx': pix_samples[:, None]})

        misc_dict = {'views': img_i,
                     'poses': pose_i,
                     'all_poses': self.all_poses}

        return in_dict, meta_dict, gt_dict, misc_dict


def chunk_lists_from_batch_reduce_to_raysamples_fn(model_input, meta, gt, max_chunk_size):

    model_in_chunked = []
    for key in model_input:
        num_views, num_rays, num_samples_per_rays, num_dims = model_input[key].shape
        chunks = torch.split(model_input[key].view(-1, num_samples_per_rays, num_dims), max_chunk_size)
        model_in_chunked.append(chunks)

    list_chunked_model_input = \
        [{k: v for k, v in zip(model_input.keys(), curr_chunks)} for curr_chunks in zip(*model_in_chunked)]

    # meta_dict
    list_chunked_zs = torch.split(meta['zs'].view(-1, num_samples_per_rays, 1),
                                  max_chunk_size)
    list_chunked_meta = [{'zs': zs} for zs in list_chunked_zs]

    # gt_dict
    gt_chunked = []
    for key in gt:
        *_, num_dims = gt[key].shape
        chunks = torch.split(gt[key].view(-1, num_dims), max_chunk_size)
        gt_chunked.append(chunks)

    list_chunked_gt = \
        [{k: v for k, v in zip(gt.keys(), curr_chunks)} for curr_chunks in zip(*gt_chunked)]

    return list_chunked_model_input, list_chunked_meta, list_chunked_gt
