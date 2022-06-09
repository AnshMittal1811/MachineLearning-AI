import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2

import numpy as np

import sys
sys.path.append('.')
from morphology import Dilation2d, Erosion2d
from basic_utils import get_gaussian_kernel

EPS = 1e-2

def add_points(positive_area, ref_images, ref_depths, ref_depth_masks, ref_poses, ref_intrinsics, min_depth=800, max_depth=1400, pts_dropout_rate=0.9, shallowest_few=-1, lindisp=False):
    # positive_area is N x 1 x H x W
    # ref_images is N x 3 x H x W
    # ref_depths is N x 1 x H x W
    # ref_depth_masks is N x 1 x H x W
    depth_levels = 100

    t_vals = torch.linspace(0., 1., steps=depth_levels).cuda()
    if not lindisp:
        depth_array = min_depth * (1. - t_vals) + max_depth * (t_vals)
    else:
        depth_array = 1. / (1. / min_depth * (1. - t_vals) + 1. / max_depth * (t_vals))

    num_views, _, H, W = positive_area.shape
    depth_array = depth_array.reshape(-1, 1, 1, 1).repeat(1, 1, H, W) # d x 1 x H x W

    unprojector = PtsUnprojector()

    all_points_to_add = []
    all_buvs = []

    do_random_dropout = pts_dropout_rate > 0.0


    for view_id in range(num_views):
        positive_mask = positive_area[view_id:view_id+1].repeat(depth_levels, 1, 1, 1) # d x 1 x H x W
        pose = ref_poses[view_id:view_id+1].repeat(depth_levels, 1, 1)
        intrinsics = ref_intrinsics[view_id:view_id + 1].repeat(depth_levels, 1, 1)

        vert_pos, buv = unprojector(depth_array, pose, intrinsics, positive_mask, return_coord=True)
        vert_pos = vert_pos.permute(1, 0) # 3xN, where N is the cartesian product of all positive rays and all possible depths
        buv[:, 0] = view_id
        buv = buv.permute(1, 0) # 3xN


        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(num_views, 1, 1)  # num_views x 4 x N, turned into homogeneous coord

        # target_pose is cam_T_world. turn into all other views
        xyz_target = ref_poses.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # num_views x 3 x N, discard homogeneous dimension

        xy_proj = ref_intrinsics.bmm(xyz_target)  # num_views x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]),
                            1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6
        # sampler is num_views x 3 x N
        # results should have shape num_views x N
        sampler = torch.round(sampler).to(torch.long)
        sampler_xs = sampler[:, 0] # num_views x N
        sampler_ys = sampler[:, 1] # num_views x N
        sampler_zs = sampler[:, 2] # num_views x N
        sampler_xs_bounded = torch.clamp(sampler_xs, 0, W-1)
        sampler_ys_bounded = torch.clamp(sampler_ys, 0, H-1)
        # do nn sample
        samples = []
        for i in range(num_views):
            sample = torch.zeros_like(zs[0]) # 1 x N

            # if the corresponding area's err is large enough, return 1
            sample += positive_area[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N

            # if out of bound, we say that view agrees
            sample[:, ((sampler_xs[i] < 0) | (sampler_xs[i] > W-1) | (sampler_ys[i] < 0) | (sampler_ys[i] > H-1))] += 1.0  # 1 x N

            # or if the sample's depth is DEEPER than the predicted depth, we also say yes.
            sample_depth = sampler_zs[i:i+1] # 1 x N
            pred_depth = ref_depths[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N
            pred_depth_mask = ref_depth_masks[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N
            pred_depth[pred_depth_mask < 0.5] = 1e6 # set to infinity, since predicetd pts in those areas are pre-filtered out

            sample[sample_depth > pred_depth] += 1.0
            sample[sample > 0.5] = 1.0

            samples.append(sample)

        samples = torch.cat(samples, dim=0) # num_views x N

        # try the simplest thing first
        pos_samples = torch.sum(samples, dim=0) == num_views # N, binary indicator

        # take the shallowest few if too many points past the test, to save resource
        if shallowest_few > 0:
            pos_samples = pos_samples.reshape(depth_levels, -1) # 100 x N
            cum_pos_samples = torch.cumsum(pos_samples, dim=0, dtype=torch.long) # accumulate the number of positive samples along the depth dim
            shallowest_samples = cum_pos_samples <= shallowest_few
            pos_samples = torch.logical_and(pos_samples.reshape(-1), shallowest_samples.reshape(-1))

        points_to_add = vert_pos[:, pos_samples] # 3 x N_valid
        buv = buv[:, pos_samples]

        if do_random_dropout:
            num_pts_to_keep = round(points_to_add.shape[1] * (1.0 - pts_dropout_rate))
            pts_id_to_keep = torch.tensor(np.random.choice(np.arange(points_to_add.shape[1]), size=num_pts_to_keep, replace=False))
            points_to_add = points_to_add[:, pts_id_to_keep]
            buv = buv[:, pts_id_to_keep]

        all_points_to_add.append(points_to_add)
        all_buvs.append(buv)


    all_points_to_add = torch.cat(all_points_to_add, dim=1) # 3 x N
    all_buvs = torch.cat(all_buvs, dim=1)  # 3 x N

    return all_points_to_add, all_buvs.to(torch.long)

def extract_error_map(model, tau_E, ref_images, ref_masks, ref_depths, ref_poses, ref_intrinsics, data_loader, dataset_args, logger):
    model.eval()

    factor = dataset_args['factor']
    render_scale = dataset_args['render_scale']

    e_op_2 = Erosion2d(1, 1, 2, soft_max=False).cuda()
    d_op_2 = Dilation2d(1, 1, 2, soft_max=False).cuda()

    with torch.no_grad():
        l1_err_maps = []
        all_imgs = []
        all_preds = []
        positive_areas = []

        for i_batch, data_blob in enumerate(data_loader):

            images, depths, poses, intrinsics = data_blob
            masks = torch.ones_like(images[:, :, 0])  # color mask

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()
            masks = masks.cuda()
            masks = masks.unsqueeze(2)

            rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1], 1 x 3 x H x W
            rgb_gt = F.interpolate(rgb_gt, [dataset_args["crop_size"][0] // (factor // render_scale),
                                            dataset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear',
                                   align_corners=True)
            mask_gt = F.interpolate(masks[:, 0], [dataset_args["crop_size"][0] // (factor // render_scale),
                                                  dataset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            intrinsics_gt = intrinsics[:, 0]
            intrinsics_gt[:, 0] /= (images.shape[3] / (dataset_args["crop_size"][0] // factor))  # rescale according to the ratio between dataset images and render images
            intrinsics_gt[:, 1] /= (images.shape[4] / (dataset_args["crop_size"][1] // factor))

            # rgb_est = model(ref_images, poses[:, 0], intrinsics_gt, is_eval=True) # 1 x 3 x H x W
            rgb_est = model.evaluate(ref_images, poses[:, 0], intrinsics_gt, num_random_samples=5)

            l1_err_map = torch.mean(torch.abs(rgb_gt - rgb_est), dim=1, keepdim=True) # 1 x 1 x H x W
            l1_err_maps.append(l1_err_map)

            mean_err = torch.mean(l1_err_map)
            std_err = torch.std(l1_err_map)
            positive_area = (l1_err_map > mean_err + tau_E * mean_err).float()
            positive_area = d_op_2(e_op_2(positive_area))  # opening

            positive_areas.append(positive_area)

            all_imgs.append(rgb_gt)
            all_preds.append(rgb_est)

        all_imgs = torch.cat(all_imgs, dim=0) # N x 3 x H x W
        all_preds = torch.cat(all_preds, dim=0) # N x 3 x H x W

        gaussian_kernel = get_gaussian_kernel(kernel_size=11, sigma=1.5).cuda()
        all_imgs = gaussian_kernel(all_imgs)
        all_preds = gaussian_kernel(all_preds)

        l1_err_maps = torch.cat(l1_err_maps, dim=0) # N x 1 x H x W
        max_err = torch.max(l1_err_maps)
        l1_err_maps_normalized = l1_err_maps / max_err # N x 1 x H x W, range[0,1]
        global_step = logger.total_steps # backup
        logger.set_global_step(0)

        positive_areas = torch.cat(positive_areas, dim=0)

        print('total number of positive pixels:', torch.sum(positive_areas))


        for i in range(l1_err_maps.shape[0]):
            logger.set_global_step(i)
            # cat the things together
            gt_to_sum = all_imgs[i:i+1] # 1 x 3 x H x W
            pred_to_sum = all_preds[i:i + 1]  # 1 x 3 x H x W
            err_to_sum = (l1_err_maps_normalized[i:i+1].repeat(1, 3, 1, 1)) * 2.0 - 1.0 # 1 x 3 x H x W, ranging [-1,1]
            pos_to_sum = (positive_areas[i:i+1].repeat(1, 3, 1, 1)) * 2.0 - 1.0  # 1 x 3 x H x W, ranging [-1,1]

            final_sum = torch.cat([gt_to_sum, pred_to_sum, err_to_sum, pos_to_sum], dim=2) # cat in vertical direction
            logger.summ_rgb('error/l1_error_map', final_sum, force_save=True)


        logger.set_global_step(global_step)

    model.train()

    return positive_areas # N x 1 x H x W

def check_depth_consistency(ref_depths, ref_poses, ref_intrinsics):
    # The Point Pruning method in the paper.
    # ref_depths is N x 1 x H x W
    # ref_depth_masks is N x 1 x H x W
    # ref_poses is N x 4 x 4

    num_views, _, H, W = ref_depths.shape
    depth_masks = torch.zeros_like(ref_depths)

    unprojector = PtsUnprojector().cuda()

    for view_id in range(num_views):
        pose = ref_poses[view_id:view_id + 1]  # 1 x 4 x 4
        intrinsics = ref_intrinsics[view_id:view_id + 1]  # 1 x 3 x 3
        depth_array = ref_depths[view_id:view_id + 1]  # 1 x 1 x H x W

        vert_pos, buv = unprojector(depth_array, pose, intrinsics, return_coord=True)
        vert_pos = vert_pos.permute(1, 0)  # 3xN, where N is the cartesian product of all positive rays and all possible depths

        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(num_views, 1,
                                                                                                     1)  # num_views x 4 x N, turned into homogeneous coord

        # target_pose is cam_T_world. turn into all other views
        xyz_target = ref_poses.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # num_views x 3 x N, discard homogeneous dimension

        xy_proj = ref_intrinsics.bmm(xyz_target)  # num_views x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]),
                            1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6
        # sampler is num_views x 3 x N
        # results should have shape num_views x N
        sampler = torch.round(sampler).to(torch.long)
        sampler_xs = sampler[:, 0]  # num_views x N
        sampler_ys = sampler[:, 1]  # num_views x N
        sampler_zs = sampler[:, 2]  # num_views x N
        sampler_xs_bounded = torch.clamp(sampler_xs, 0, W - 1)
        sampler_ys_bounded = torch.clamp(sampler_ys, 0, H - 1)
        # do nn sample
        samples = []
        for i in range(num_views):
            sample_depth = sampler_zs[i:i + 1]  # 1 x N
            pred_depth = ref_depths[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]]  # 1 x N

            sample = torch.zeros_like(sample_depth)  # 1 x N
            # if out of bound, we say that view agrees since it doesn't know
            sample[:, ((sampler_xs[i] < 0) | (sampler_xs[i] > W - 1) | (sampler_ys[i] < 0) | (
                        sampler_ys[i] > H - 1))] = 1.0  # 1 x N

            sample[sample_depth >= .8 * pred_depth] = 1.0

            samples.append(sample.reshape(1, H, W))

        samples = torch.cat(samples, dim=0)  # n_views x H x W
        samples = (torch.sum(samples, dim=0) >= num_views - 1).float()  # H x W, only 1 if all 1s, i.e. the predicted depth don't block ANY other view

        depth_masks[view_id] = samples

    return depth_masks


def get_view_dir_world(ref_pose, view_dir_cam):
    view_dir = (torch.inverse(ref_pose) @ view_dir_cam.view(1, 4, 1))  # B x 4 x 1
    view_dir = view_dir[:, 0:3]  # B x 3 x 1
    view_dir = view_dir.repeat(1, 1, xyz_ndc.shape[1])  # B x 3 x N
    # view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3

    return view_dir

def get_view_dir_world_per_ray(ref_pose, view_dir_cam):
    # view_dir_cam is B x 3 x N
    view_dir_cam = view_dir_cam / torch.linalg.norm(view_dir_cam, dim=1, keepdim=True) # B x 3 x N
    view_dir_cam = torch.cat([view_dir_cam, torch.zeros_like(view_dir_cam[:, 0:1])], dim=1) # B x 4 x N. turn into homogeneous coord. last dim zero because we want dir only
    view_dir = (torch.inverse(ref_pose) @ view_dir_cam) # B x 4 x N
    view_dir = view_dir[:, 0:3]  # B x 3 x N

    return view_dir

def crop_operation(images, intrinsics, crop_h, crop_w, mod='random'):
    B, _, H, W = images.shape
    # concat all things together on feat dim if you want to crop, say, depth as well.

    new_images = []
    new_intrinsics = []

    for b in range(B):
        if mod == 'random':
            x0 = np.random.randint(0, W - crop_w + 1)
            y0 = np.random.randint(0, H - crop_h + 1)
        elif mod == 'center':
            x0 = (wd1 - crop_w) // 2
            y0 = (ht1 - crop_h) // 2
        else:
            raise NotImplementedError

        x1 = x0 + crop_w
        y1 = y0 + crop_h
        new_image = images[b, :, y0:y1, x0:x1]
        new_intrinsic = torch.clone(intrinsics[b])

        new_intrinsic[0, 2] -= x0
        new_intrinsic[1, 2] -= y0

        new_images.append(new_image)
        new_intrinsics.append(new_intrinsic)

    new_images = torch.stack(new_images, dim=0)
    new_intrinsics = torch.stack(new_intrinsics, dim=0)

    return new_images, new_intrinsics

class PtsUnprojector(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(PtsUnprojector, self).__init__()
        self.device=device

    def forward(self, depth, pose, intrinsics, mask=None, return_coord=False):
        # take depth and convert into world pts
        # depth: B x 1 x H x W
        # pose: B x 4 x 4
        # intrinsics: B x 3 x 3
        # mask: B x 1 x H x W
        # return coord: return the corresponding [b,y,x] coord for each point, so that we can index into the vertex feature

        B, _, H, W = depth.shape

        # assert(h==self.H)
        # assert(w==self.W)
        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1).to(self.device)  # 1 x 3 x N

        depth = depth.reshape(B, 1, -1)

        projected_coors = xyzs * depth # B x 3 x N

        xyz_source = torch.inverse(intrinsics).bmm(projected_coors)  # B x 3 x N, xyz in cam1 space
        xyz_source = torch.cat((xyz_source, torch.ones_like(xyz_source[:, 0:1])), dim=1) # B x 4 x N

        # pose is cam_T_world
        xyz_world = torch.inverse(pose).bmm(xyz_source) # B x 4 x N
        xyz_world = xyz_world[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        # xyz_world = xyz_world.reshape(B, 3, H, W)
        xyz_world = xyz_world.permute(0, 2, 1).reshape(-1, 3)  # B*N x 3

        if return_coord:
            bs = torch.linspace(0, B-1, B).float()
            bs = bs.view(B, 1, 1).repeat(1, H, W)
            xs = xs.view(1, H, W).repeat(B, 1, 1)
            ys = ys.view(1, H, W).repeat(B, 1, 1)

            buvs = torch.stack((bs, ys, xs), dim=-1).view(-1, 3).to(self.device) # B*N x 3

        # if mask not none, we prune the xyzs by only selecting the valid ones
        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask>0.5)[0]
            xyz_world = xyz_world[nonzeros, :] # n_valid x 3
            if return_coord:
                buvs = buvs[nonzeros, :]

        if return_coord:
            return xyz_world, buvs.to(torch.long)
        else:
            return xyz_world

    def get_dists(self, depth, intrinsics, mask=None):
        # take depth and convert into world pts
        # depth: B x 1 x H x W
        # intrinsics: B x 3 x 3
        # mask: B x 1 x H x W
        B, _, H, W = depth.shape

        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1).to(self.device)  # 1 x 3 x N

        depth = depth.reshape(B, 1, -1)

        projected_coors = xyzs * depth  # B x 3 x N

        xyz_source = torch.inverse(intrinsics).bmm(projected_coors)  # B x 3 x N, xyz in cam1 space

        l2_dists =  torch.norm(xyz_source, p=2, dim=1, keepdim=True)  # B x 1 x N
        l2_dists = l2_dists.permute(0, 2, 1).reshape(-1, 1)  # B*N x 1

        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask > 0.5)[0]
            l2_dists = l2_dists[nonzeros, :]  # n_valid x 3

        return l2_dists

    def apply_mask(self, feat, mask):
        # feat: B x C x H x W
        # mask: B x 1 x H x W
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, -1)
        feat = feat.permute(0, 2, 1).reshape(-1, C)  # B*N x C

        mask = mask.reshape(-1)
        nonzeros = torch.where(mask > 0.5)[0]
        feat = feat[nonzeros, :]  # n_valid x C

        return feat


# adapted from barf:
# https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/f04e37cc3417bab25d71ccd9146bf696534ff3b1/camera.py#L83
class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,6]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans


#### adapted from NeRF. for computing animation pose
def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), [0, 0, 0, 1.]], axis=0)) # 4 x 4

    return render_poses

def get_animation_poses(ref_poses):
    # ref_poses is N x 4 x 4, w2c
    w2c = ref_poses.cpu().numpy() # to np
    c2w = np.linalg.inv(w2c) # N x 4 x 4

    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)

    focal = 800 # DTU

    avg_pose = poses_avg(c2w)

    N_views = 120
    N_rots = 2

    render_poses = render_path_spiral(avg_pose, up, rads, focal, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_poses = np.linalg.inv(render_poses)

    return torch.tensor(render_poses).cuda()



