from __future__ import print_function, division
import sys
sys.path.append('core')
sys.path.append('datasets')

import argparse
import os
import time
import json
import glob
# import matplotlib.pyplot as plt

from dtu import DTUViewsynTrain
from llff import LLFFViewsynTrain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvF

from projector import Shader
from modules.unet import SmallUNet, UNet, TwoLayersCNN
import frame_utils

from basic_utils import smoothnessloss, fetch_optimizer, sequence_loss_rgb
from geom_utils import check_depth_consistency, get_view_dir_world_per_ray, PtsUnprojector, add_points, extract_error_map
from summ_utils import Logger

from torch.cuda.amp import GradScaler

import lpips

from pytorch3d.structures import Pointclouds

# from pytorch3d.renderer import (
#     FoVOrthographicCameras,
#     PointsRasterizationSettings,
#     PointsRasterizer,
#     PulsarPointsRenderer,
# )

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer
)

from pulsar.unified import PulsarPointsRenderer

EPS = 1e-2

def validate(model, ref_images, val_loader, valset_args, logger, pts_to_use_list=None, rasterize_rounds=5):
    model.eval()
    metrics = {}

    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    # lpips_vgg = None

    total_render_time = 0.0

    with torch.no_grad():
        for i_batch, data_blob in enumerate(val_loader):
            images, _, poses, intrinsics = data_blob
            masks = torch.ones_like(images[:, :, 0]) # color mask

            factor = valset_args['factor']
            render_scale = valset_args['render_scale']
            loss_type = valset_args['loss_type']

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()
            masks = masks.cuda()
            masks = masks.unsqueeze(2)

            rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1]
            rgb_gt = F.interpolate(rgb_gt, [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear',
                                   align_corners=True)
            mask_gt = F.interpolate(masks[:, 0], [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            intrinsics_gt = intrinsics[:, 0]
            intrinsics_gt[:, 0] /= (images.shape[3] / (valset_args["crop_size"][0] // factor))  # rescale according to the ratio between dataset images and render images
            intrinsics_gt[:, 1] /= (images.shape[4] / (valset_args["crop_size"][1] // factor))

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_pose = poses[:, 0] # B x 4 x 4

            start.record()
            rgb_est = [model.evaluate(ref_images, target_pose, intrinsics_gt, num_random_samples=rasterize_rounds, pts_to_use_list=pts_to_use_list), ]  # 1 x 3 x H x W
            end.record()

            torch.cuda.synchronize()

            print('total render time for one image:', start.elapsed_time(end))\

            total_render_time += start.elapsed_time(end)

            _, rgb_metrics = sequence_loss_rgb(rgb_est, rgb_gt, mask_gt, lpips_vgg=lpips_vgg,
                                                      loss_type=loss_type,
                                                      weight=None,
                                                      gradual_weight=None)

            if len(metrics) == 0: # init
                metrics.update(rgb_metrics)
            else: # update
                for (k, v) in metrics.items():
                    metrics[k] += rgb_metrics[k]

            print('finished rendering %d/%d' % (i_batch+1, len(val_loader)))
            print(rgb_metrics)

            logger.summ_rgb('eval/rgb_gt/%d' % i_batch, rgb_gt, mask_gt, force_save=True)
            logger.summ_rgb('eval/rgb_est/%d' % i_batch, rgb_est[-1], mask_gt, force_save=True)
            logger.summ_diff('eval/l1_diff/%d' % i_batch, rgb_gt, rgb_est[-1], force_save=True)

    # average
    for (k, v) in metrics.items():
        metrics[k] /= len(val_loader)

    # compute the "avg. metric" from mipnerf
    avg = (10.**(-metrics['psnr'] / 10.) * metrics['lpips'] * np.sqrt(1-metrics['ssim'])) ** (1./3.)
    metrics['avg'] = avg

    logger.write_dict(metrics, 'eval')

    print('finish eval on %d samples' % len(val_loader))
    print(metrics)

    print('average render time: %.1f' % (total_render_time / len(val_loader)))

    model.train()

    return metrics


class PulsarSceneModel(nn.Module):
    def __init__(self,
                 vert_pos,
                 dim_pointfeat=256,
                 radius=7.5e-4,
                 render_size=(300, 400),
                 world_scale=400.,
                 render_scale=1,
                 bkg_col=(0,0,0),
                 gamma=1.0e-3,
                 free_xyz=False,
                 free_opy=False,
                 free_rad=False,
                 do_2d_shading=False,
                 shader_arch='simple_unet',
                 pts_dropout_rate=0.0,
                 basis_type='mlp',
                 shader_output_channel=128,
                 shader_norm='none',
                 ):
        super(PulsarSceneModel, self).__init__()
        # images: N x 3 x H x W
        # depth_low_res: N x 1 x h x w
        # masks_low_res: N x 1 x h x w

        self.free_opy = free_opy
        self.free_xyz = free_xyz
        self.free_rad = free_rad


        self.unprojector = PtsUnprojector()

        if args.free_xyz:
            self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=True))
        else:
            self.register_buffer('vert_pos', vert_pos)  # 3 x N


        self.n_points = vert_pos.shape[1]

        device = torch.device("cuda")


        if basis_type == 'mlp':
            self.register_parameter("vert_feat", nn.Parameter(torch.randn(self.n_points, dim_pointfeat), requires_grad=True))
        elif basis_type=='SH':
            self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat), requires_grad=True))
        elif basis_type=='none':
            self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat), requires_grad=True))
        else:
            raise NotImplementedError


        z_dir = torch.tensor([0, 0, 1, 0]).reshape(1, 4).float()  # the last element is 0, because we only care orientation
        self.register_buffer("z_dir", z_dir)

        if do_2d_shading:
            print('using shader arch:', shader_arch)
            self.shader_output_channel = shader_output_channel
            assert shader_arch == 'simple_unet' # other render_scale option not supported yet

            if shader_arch == 'simple_unet':
                self.shader_2d = SmallUNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm, render_scale=render_scale)
            elif shader_arch == 'full_unet':
                self.shader_2d = UNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm)
            elif shader_arch == 'simple':
                self.shader_2d = TwoLayersCNN(n_channels=self.shader_output_channel, n_classes=3, norm=shader_norm)
            else:
                raise NotImplementedError

        else:
            self.shader_output_channel = 3 # override

        output_opacity = not free_opy # if not free, ouput opacity from the network

        # self.shader = Shader(feat_dim=dim_pointfeat, rgb_channel=self.shader_output_channel, output_opacity=self.free_opy, opacity_channel=1)
        self.shader = Shader(feat_dim=dim_pointfeat, rgb_channel=self.shader_output_channel, output_opacity=output_opacity, opacity_channel=1, basis_type=basis_type)

        if free_rad:
            raise NotImplementedError
        else:
            pass

        if free_opy:
            self.register_parameter("vert_opy", nn.Parameter(torch.ones(self.n_points), requires_grad=True))
        else:
            raise NotImplementedError
        
        cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32, device=device)[None, ...]),
                                         T=torch.zeros((1, 3), dtype=torch.float32, device=device),
                                         znear=[1.0],
                                         zfar=[1e5],
                                         device=device,
                                         )

        raster_settings = PointsRasterizationSettings(
                image_size=render_size,
                radius=None,
                max_points_per_bin=50000
            )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.renderer = PulsarPointsRenderer(rasterizer=rasterizer, max_num_spheres=vert_pos.shape[1], n_channels=self.shader_output_channel, n_track=100).cuda()

        if self.shader_output_channel==3:
            self.register_buffer('bkg_col', torch.tensor(bkg_col, dtype=torch.float32, device=device))
        else: 
            # high-dim feature vector
            self.register_parameter('bkg_col', nn.Parameter(torch.randn(self.shader_output_channel, dtype=torch.float32, device=device), requires_grad=True))

        self.render_size = render_size
        self.gamma = gamma
        self.dim_pointfeat = dim_pointfeat

        self.H, self.W = render_size[0], render_size[1]
        self.radius = radius

        self.do_2d_shading = do_2d_shading
        self.pts_dropout_rate = pts_dropout_rate
        self.world_scale = world_scale

    def forward(self, target_pose, target_intrinsics, affine_params=None, is_eval=False):
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3
        # affine_params: for data agumentation. not used in this paper
        # if is_eval: turn off random dropout.

        do_random_dropout = ((not is_eval) and (self.pts_dropout_rate > 0.0))
        if do_random_dropout:
            num_pts_to_keep = round(self.vert_pos.shape[1] * (1.0 - self.pts_dropout_rate))
            pts_id_to_keep = torch.multinomial(torch.ones_like(self.vert_pos[0]), num_pts_to_keep, replacement=False) # this version is much faster

        B = target_pose.shape[0]

        # convert self.vert_pos in world coordinates into cam coordinates
        xyz_world = torch.cat((self.vert_pos, torch.ones_like(self.vert_pos[0:1])), dim=0).unsqueeze(0).repeat(B, 1, 1)  # 1 x 4 x N, turned into homogeneous coord

        # target_pose is world2cam
        xyz_target = target_pose.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        xy_proj = target_intrinsics.bmm(xyz_target) # B x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6

        # compute the radius based on the distance of the points to the reference view camera center
        scale_pts = torch.norm(sampler, dim=1) / self.world_scale
        radius = scale_pts.detach() * self.radius # B x N. detaching here avoids a backward problem
        if do_random_dropout:
            radius = radius[:, pts_id_to_keep]

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3


        if do_random_dropout:
            xyz_ndc = xyz_ndc[:, pts_id_to_keep] # B x N_drop x 3

        # do shading
        pointfeat = self.vert_feat # N x feat_dim

        if do_random_dropout:
            pointfeat = pointfeat[pts_id_to_keep]

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim
        view_dir = get_view_dir_world_per_ray(target_pose, xyz_target.detach()) # B x 3 x N
        if do_random_dropout:
            view_dir = view_dir[:, :, pts_id_to_keep]

        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3

        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3

        if not self.free_opy: # opy from the network
            shaded_feature, shaded_opy = torch.split(shaded_feature, [self.shader_output_channel, 1], dim=1)
            shaded_opy = shaded_opy.reshape(B, xyz_ndc.shape[1]) # B x N

        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N

        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)

        # the pulsar NDC space needs this scaling
        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))
        pts3D = Pointclouds(points=xyz_ndc, features=shaded_feature.permute(0, 2, 1))

        if self.free_opy:
            opacity = torch.sigmoid(self.vert_opy.unsqueeze(0).repeat(B, 1))
            if do_random_dropout:
                opacity = opacity[:, pts_id_to_keep]
        else:
            # already dropout before feeding into the net, so no need to do here again.
            opacity = torch.sigmoid(shaded_opy) # B x N


        pred = self.renderer(
            pts3D,
            radius=radius,
            gamma=[self.gamma] * B,  # Renderer blending parameter gamma, in [1., 1e-5].
            znear=[1.0] * B,
            zfar=[1e5] * B,
            radius_world=True,
            bg_col=self.bkg_col,
            opacity=torch.clamp(opacity, 0.0, 1.0)
        )
        # pred: B x H x W x 3
        pred = pred.permute(0, 3, 1, 2).contiguous() # B x 3 x H x W


        if affine_params is not None:
            # crop before the shader2d
            pred = tvF.affine(pred, *affine_params, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        if self.do_2d_shading: # do post-processing
            rgb_pred = self.shader_2d(pred)
        else:
            rgb_pred = pred

        return pred, rgb_pred

    def evaluate(self, ref_images, target_pose, target_intrinsics, num_random_samples=5, pts_to_use_list=None, target_viewpose=None):
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3
        # this fucntion support multiple rasterization rounds

        # for making the animation with fixed cam and varying lighting
        # else we keep view dir and cam dir the same
        if target_viewpose is None: 
            target_viewpose = target_pose
            
        do_random_dropout = (self.pts_dropout_rate > 0.0)
        if not do_random_dropout:
            num_random_samples = 1

        vert_pos = self.vert_pos
        vert_feat = self.vert_feat
        vert_opy = self.vert_opy

        if pts_to_use_list is not None:
            vert_pos = vert_pos[:, pts_to_use_list]
            vert_feat = vert_feat[pts_to_use_list, :]
            vert_opy = vert_opy[pts_to_use_list]

        num_pts_to_keep = round(self.vert_pos.shape[1] * (1.0 - self.pts_dropout_rate))

        B = target_pose.shape[0]


        # convert self.vert_pos in world coordinates into cam coordinates
        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(B, 1, 1)  # 1 x 4 x N, turned into homogeneous coord

        # tagget_pose is cam_T_world
        xyz_target = target_pose.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        xy_proj = target_intrinsics.bmm(xyz_target) # B x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6

        # compute the radius based on the distance of the points to the reference view camera center
        scale_pts = torch.norm(sampler, dim=1) / self.world_scale
        radius = scale_pts * self.radius # B x N

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3

        # do shading
        pointfeat = vert_feat # N x feat_dim

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim

        # view_dir = get_view_dir_world_per_ray(target_pose, xyz_target)
        view_dir = get_view_dir_world_per_ray(target_viewpose, xyz_target)
        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3
        # view_dir = get_view_dir_world(target_pose, self.z_dir)

        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3
        
        if not self.free_opy: # opy from the network
            shaded_feature, shaded_opy = torch.split(shaded_feature, [self.shader_output_channel, 1], dim=1)
            shaded_opy = shaded_opy.reshape(B, xyz_ndc.shape[1]) # B x N

        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N


        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)


        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))

        if self.free_opy:
            opacity = torch.sigmoid(vert_opy.unsqueeze(0).repeat(B, 1))
        else:
            # already dropout before feeding into the net, so no need to do here again.
            opacity = torch.sigmoid(shaded_opy) # B x N


        # sample the rasterization step multiple times to get better results.
        all_preds = None
        for _ in range(num_random_samples):
            pts_id_to_keep = torch.multinomial(torch.ones_like(vert_pos[0]), num_pts_to_keep, replacement=False) # this version is much faster.

            xyz_ndc_sampled = xyz_ndc[:, pts_id_to_keep]  # B x N_drop x 3
            shaded_feature_sampled = shaded_feature[..., pts_id_to_keep]
            shaded_opy_sampled = opacity[:, pts_id_to_keep]
            radius_sampled = radius[:, pts_id_to_keep]

            pts3D = Pointclouds(points=xyz_ndc_sampled, features=shaded_feature_sampled.permute(0, 2, 1))

            pred = self.renderer(
                pts3D,
                radius=radius_sampled,
                gamma=[self.gamma] * B,  # Renderer blending parameter gamma, in [1., 1e-5].
                znear=[1.0] * B,
                zfar=[1e5] * B,
                radius_world=True,
                bg_col=self.bkg_col,
                opacity=torch.clamp(shaded_opy_sampled, 0.0, 1.0)
            )
            # pred: B x H x W x 3
            pred = pred.permute(0, 3, 1, 2).contiguous()  # B x 3 x H x W

            if all_preds is None:
                all_preds = pred
            else:
                all_preds += pred

         
        pred = all_preds / float(num_random_samples)

        if self.do_2d_shading: # do post-processing
            rgb_pred = self.shader_2d(pred)

        else:
            rgb_pred = pred

        return rgb_pred

def train(args):
    params = {}
    for k in list(vars(args).keys()):
        params[k] = vars(args)[k]

    if args.tb_log_dir is not None:
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.name)
        if not os.path.isdir(args.tb_log_dir):
            os.mkdir(args.tb_log_dir)

    with open('dir.json') as f:
        d = json.load(f)

    d = d[args.setting]

    HR = params["HR"]
    factor = 8 if not HR else 4

    unprojector = PtsUnprojector()

    render_scale = args.render_scale

    # extract pts for all views
    gpuargs = {'num_workers': 0, 'drop_last': False, 'shuffle': False}

    datasetname = d["dataset"]
    trainset_args = {"num_frames": 1,
                     "crop_size": [args.crop_h, args.crop_w],
                     # "resize": [args.resize_h, args.resize_w]
                     "resize": [args.crop_h, args.crop_w],
                     "precomputed_depth_path": args.precomputed_depth_path,
                     "single": args.single
                     }

    if datasetname == "LLFF":
        total_num_views = len(sorted(glob.glob(os.path.join(d["testing_dir"], args.single, "DTU_format", "images", "*.jpg"))))

        indicies = np.arange(total_num_views)

        trainset_args["data_augmentation"] = False
        trainset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 8) != 0]) # this is the same as NeRF

    elif datasetname == "DTU":
        trainset_args["return_mask"] = False

        indicies = np.arange(49)

        trainset_args["source_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) != 2] # keep every 7-th image as the test view
        trainset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) != 2]

    else:
        raise NotImplementedError

    valset_args = trainset_args.copy()

    if datasetname == "LLFF":
        valset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 8) == 0]) # this is the same as NeRF
    elif datasetname == "DTU":
        valset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) == 2] # keep every 7-th image as the test view

    # turn off random scale and crop
    valset_args["crop_size"] = [args.crop_h, args.crop_w]
    valset_args["resize"] = [args.crop_h, args.crop_w]

    train_dataset = eval(datasetname+'ViewsynTrain')(d["testing_dir"], **trainset_args)
    val_dataset = eval(datasetname+'ViewsynTrain')(d["testing_dir"], **valset_args)

    valset_args['factor'] = factor
    valset_args['render_scale'] = render_scale
    valset_args['loss_type'] = args.loss_type

    train_loader = DataLoader(train_dataset, batch_size=1, **gpuargs)
    val_loader = DataLoader(val_dataset, batch_size=1, **gpuargs)


    trainset_images = []
    trainset_depth_masks = []
    trainset_loss_masks = []
    trainset_depths = []
    trainset_poses = []
    trainset_intrinsics = []

    # put all training samples into memory
    for _, data_blob in enumerate(train_loader):

        images, depths, poses, intrinsics = data_blob
        loss_masks = torch.ones_like(images[:, :, 0]) # everything

        # now just simple filtering. later we may use more clever pre-filtering
        if datasetname == 'LLFF':
            depth_masks = (depths > 400.0).float()
        elif datasetname == 'DTU':
            depth_masks = (depths > 400.0).float() * (depths < 1400.0).float()
        else:
            raise NotImplementedError


        images = images.cuda()
        poses = poses.cuda()
        intrinsics = intrinsics.cuda()
        loss_masks = loss_masks.cuda()
        depth_masks = depth_masks.cuda()
        depths = depths.cuda()
        depth_low_res = F.interpolate(depths, [params['crop_h'] // factor, params['crop_w'] // factor], mode='nearest')

        depths = depths.unsqueeze(2)
        depth_low_res = depth_low_res.unsqueeze(2) # 1 x 1 x H x W
        loss_masks = loss_masks.unsqueeze(2)
        depth_masks = depth_masks.unsqueeze(2)

        rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1], 1 x H x W
        rgb_gt = F.interpolate(rgb_gt, [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='bilinear', align_corners=True)
        loss_mask_gt = F.interpolate(loss_masks[:, 0], [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='nearest') # B x 1 x H x W
        depth_mask_gt = F.interpolate(depth_masks[:, 0], [params['crop_h'] // factor, params['crop_w'] // factor], mode='nearest')  # B x 1 x H x W

    
        intrinsics_gt = intrinsics[:, 0] # B x 4 x 4
        intrinsics_gt[:, 0] /= (images.shape[3] / (params['crop_h'] // factor)) # rescale according to the ratio between dataset images and render images
        intrinsics_gt[:, 1] /= (images.shape[4] / (params['crop_w'] // factor))

        trainset_images.append(rgb_gt)
        trainset_depth_masks.append(depth_mask_gt)
        trainset_loss_masks.append(loss_mask_gt)
        trainset_depths.append(depth_low_res[:, 0])
        trainset_poses.append(poses[:, 0])
        trainset_intrinsics.append(intrinsics_gt)


    # stack
    trainset_images = torch.cat(trainset_images, dim=0)
    trainset_depth_masks = torch.cat(trainset_depth_masks, dim=0)
    trainset_loss_masks = torch.cat(trainset_loss_masks, dim=0)
    trainset_depths = torch.cat(trainset_depths, dim=0)
    trainset_poses = torch.cat(trainset_poses, dim=0)
    trainset_intrinsics = torch.cat(trainset_intrinsics, dim=0)

    if args.do_check_depth_consistency:
        # do point pruning

        consistency_induced_depth_masks = check_depth_consistency(trainset_depths, trainset_poses, trainset_intrinsics) # N x 1 x H x W
        trainset_depth_masks_original = trainset_depth_masks.clone()
        trainset_depth_masks = trainset_depth_masks * consistency_induced_depth_masks
        pruned_points_mask = trainset_depth_masks_original - trainset_depth_masks

        # # for paper visualization
        # pts_before_prune = int(torch.sum(trainset_depth_masks_original).item())
        # print('number of points before pruning: %d' % pts_before_prune)
        # pts_after_prune = int(torch.sum(trainset_depth_masks).item())
        # print('number of points after pruning: %d' % pts_after_prune)
        # print('number of points pruned: %d' % (pts_before_prune - pts_after_prune))
    
        # pruned_xyzs = []

        for i in range(len(trainset_images)):
            xyzs, _ = unprojector(trainset_depths[i:i + 1], trainset_poses[i:i + 1], trainset_intrinsics[i:i + 1], mask=pruned_points_mask[i:i + 1], return_coord=True)  # N x 3
            # pruned_xyzs.append(xyzs)

        # pruned_xyzs = torch.cat(pruned_xyzs, dim=0)  # N x 3
        # color_pruned = torch.tensor([1, 0, 0]).reshape(1, 3).repeat(pruned_xyzs.shape[0], 1)  # blue
        # frame_utils.save_ply('./pointclouds/pruned_%s.ply' % args.name, pruned_xyzs, color_pruned)



    trainset_xyzs = []
    trainset_buvs = []

    for i in range(len(trainset_images)):
        xyzs, buvs = unprojector(trainset_depths[i:i+1], trainset_poses[i:i+1], trainset_intrinsics[i:i+1], mask=trainset_depth_masks[i:i+1], return_coord=True)  # N x 3
        buvs[:, 0] = i

        trainset_xyzs.append(xyzs)
        trainset_buvs.append(buvs)

    trainset_xyzs = torch.cat(trainset_xyzs, dim=0) # N x 3
    trainset_buvs = torch.cat(trainset_buvs, dim=0) # N x 3. these are cooresponding to xyzs, so that we can index into the images/feature for each point

    if args.restore_pointclouds is None:
        vert_pos = trainset_xyzs.permute(1,0) # 3xN
        buvs = trainset_buvs.permute(1,0) # 3xN
    else:
        print('loading points from %s' % args.restore_pointclouds)
        if args.restore_pointclouds.endswith(".pt"):
            tmp = torch.load(args.restore_pointclouds)
            vert_pos = tmp['xyzs'].permute(1,0) # 3xN
            buvs = tmp['buvs'].permute(1,0) # 3xN
        elif args.restore_pointclouds.endswith(".ply"):
            vert_pos = frame_utils.load_ply(args.restore_pointclouds) # N x 3
            vert_pos = torch.tensor(vert_pos).permute(1,0) # 3xN
            buvs = None
        else:
            raise NotImplementedError

    print('total points we gonna use: %d' % vert_pos.shape[1])

    max_num_pts = args.max_num_pts

    if vert_pos.shape[1] > max_num_pts:
        print('Random dropping points to %d...' % max_num_pts)
        vert_id_to_keep = np.random.choice(np.arange(vert_pos.shape[1]), size=max_num_pts, replace=False)
        vert_pos = vert_pos[:, vert_id_to_keep]


    # PointAdd params
    tau_E = 4.0
    shallowest_few = 5
    pointadd_dropout = 0.0

    if datasetname == "LLFF":
        bkg_col = (-1, -1, 1)
        min_depth = 800
        max_depth = 1e4 # this number can be arbitrarily large, as we will sample in the disparity space
        lindisp=True

    elif datasetname == "DTU":
        bkg_col = (-1, -1, -1)
        min_depth = 800
        max_depth = 1400
        lindisp = False

    else:
        raise NotImplementedError

    model = PulsarSceneModel(vert_pos=vert_pos, dim_pointfeat=args.dim_pointfeat, render_size=(params['crop_h'] // factor, params['crop_w'] // factor),
                             render_scale=render_scale, gamma=args.blend_gamma, radius=args.sphere_radius,
                             free_xyz=args.free_xyz, free_opy=args.free_opy, free_rad=args.free_rad, bkg_col=bkg_col,  # green for debugging, red for llff
                             do_2d_shading=args.do_2d_shading, shader_arch=args.shader_arch, pts_dropout_rate=args.pts_dropout_rate,
                             basis_type=args.basis_type, shader_output_channel=args.shader_output_channel, shader_norm=args.shader_norm).cuda()

    if args.restore_ckpt is not None:
        tmp = torch.load(args.restore_ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=False)

    if args.freeze_shader:
        for param in model.shader.parameters():
            param.requires_grad = False

    # optimizer
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=True)
    logger = Logger(model, scheduler, args.outputfile, args.SUM_FREQ, args.IMG_LOG_FREQ, args.tb_log_dir)

    # make an animation and collect test-set statistics 
    if args.render_only:
        assert (args.restore_ckpt is not None)
        pts_to_use_list=None
        validate(model, trainset_images, val_loader, valset_args, logger, pts_to_use_list, rasterize_rounds=args.rasterize_rounds)
        frame_utils.make_animation(model, args.name, val_dataset, trainset_intrinsics, logger, rasterize_rounds=1)
        
        logger.close()
        return

    if args.pointadd_only:
        assert (args.restore_ckpt is not None)

        positive_area = extract_error_map(model, tau_E, trainset_images, trainset_depth_masks, trainset_depths, trainset_poses, trainset_intrinsics, train_loader, valset_args, logger)

        points_added, buvs_added = add_points(positive_area, trainset_images, trainset_depths, trainset_depth_masks, trainset_poses, trainset_intrinsics, min_depth=min_depth, max_depth=max_depth, pts_dropout_rate=pointadd_dropout, shallowest_few=shallowest_few, lindisp=lindisp)

        points_added = points_added.permute(1,0)  # N x 3
        buvs_added = buvs_added.permute(1,0)  # N x 3

        print('number of points to add: %d' % points_added.shape[0])
        color_added = torch.tensor([0, 0, 1]).reshape(1, 3).repeat(points_added.shape[0], 1) # red

        points_original = trainset_xyzs # N x 3
        color_orginal = trainset_images[trainset_buvs[:,0], :, trainset_buvs[:,1], trainset_buvs[:,2]].cpu() # N x 3

        frame_utils.save_ply('./pointclouds/%s.ply' % args.name, torch.cat([points_added, points_original]), torch.cat([color_added, color_orginal]))

        tmp = {'xyzs': torch.cat([points_added, points_original]).cpu(), 'buvs': torch.cat([buvs_added, trainset_buvs]).cpu()} # N x 3
        torch.save(tmp, './pointclouds/%s.pt' % args.name)

        logger.close()
        return

    VAL_FREQ = args.VAL_FREQ

    tic = None
    total_time = 0

    best_score = 1.0 # this is high enough

    for total_steps in range(1, args.num_steps+1):

        optimizer.zero_grad()

        train_id = np.random.choice(np.arange(len(trainset_images)), args.batch_size, replace=False)

        rgb_gt = trainset_images[train_id] # b x 3 x H x W
        target_pose = trainset_poses[train_id] # b x 4 x 4
        target_intrinsics = trainset_intrinsics[train_id] # b x 3 x 3
        mask_gt = trainset_loss_masks[train_id] # b x 1 x H x W

        if args.do_random_affine:
            affine_params = tvT.RandomAffine(0).get_params(degrees=(-30, 30), translate=(0.1, 0.1), scale_ranges=(0.9, 1.1), shears=None, img_size=(rgb_gt.shape[-1], rgb_gt.shape[-2]))
            rgb_gt, mask_gt = tvF.affine(rgb_gt, *affine_params, interpolation=torchvision.transforms.InterpolationMode.BILINEAR), tvF.affine(mask_gt, *affine_params, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        else:
            affine_params = None

        feat_est, rgb_est = model(target_pose, target_intrinsics, affine_params)  # b x 3 x H x W

        loss_type = args.loss_type

        loss, metrics = sequence_loss_rgb([rgb_est,], rgb_gt, mask_gt, loss_type=loss_type)
    
        # add the TV loss here
        if args.feat_smooth_loss_coeff > 0.0:
            if render_scale != 1:
                ht, wd = rgb_est.shape[-2:]
                feat_est = F.interpolate(feat_est, [ht, wd], mode='bilinear', align_corners=True)

            feat_smooth_loss = smoothnessloss(feat_est, mask_gt)
            loss += args.feat_smooth_loss_coeff * feat_smooth_loss
            metrics['unscaled_feat_smooth_loss'] = feat_smooth_loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()

        logger.push(metrics, 'train')
        logger.summ_rgb('train/rgb_gt_unmasked', rgb_gt)
        logger.summ_rgb('train/rgb_gt', rgb_gt, mask_gt)
        logger.summ_rgb('train/rgb_est', rgb_est, mask_gt)
        logger.summ_rgb('train/rgb_est_unmasked', rgb_est)

        if total_steps % VAL_FREQ == VAL_FREQ - 1:
            res = validate(model, trainset_images, val_loader, valset_args, logger, rasterize_rounds=args.rasterize_rounds)
            cur_score = res['avg']
            if cur_score < best_score: # best. avg the lower the better
                best_score = cur_score
                PATH = 'checkpoints/model_best_%s.pth' % args.name
                torch.save(model.state_dict(), PATH)

        logger.set_global_step(total_steps)

        if not tic is None:
            total_time += time.time() - tic
            print(
                f"time per step: {total_time / (total_steps - 1)}, expected: {total_time / (total_steps - 1) * args.num_steps / 24 / 3600} days")
            print(args.name)
        tic = time.time()

    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    # evaluate the best model
    tmp = torch.load('checkpoints/model_best_%s.pth' % args.name)
    model.load_state_dict(tmp, strict=False)
    validate(model, trainset_images, val_loader, valset_args, logger, rasterize_rounds=args.rasterize_rounds)

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' training args'''
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--restore_pointclouds', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--SUM_FREQ', type=int, default=100)
    parser.add_argument('--VAL_FREQ', type=int, default=5000)
    parser.add_argument('--IMG_LOG_FREQ', type=int, default=100)  # tensorboard log dir
    parser.add_argument('--outputfile', type=str, default=None)  # in case stdoutput is buffered (don't know how to disable buffer...)
    parser.add_argument('--tb_log_dir', type=str, default=None)  # tensorboard log dir
    parser.add_argument('--pointadd_only', type=int, default=False)
    parser.add_argument('--render_only', type=int, default=False)

    '''loss args'''
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--loss_type', type=str, default='l1')
    parser.add_argument('--feat_smooth_loss_coeff', type=float, default=0.0)
    parser.add_argument(
        '--special_args_dict',
        type=lambda x: {k: float(v) for k, v in (i.split(':') for i in x.split(','))},
        default={},
        help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9'
    )
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--pct_start', type=float, default=0.001)

    '''dataset args'''
    parser.add_argument('--setting', type=str, default='DTU')
    parser.add_argument('--crop_h', type=int, default=448)
    parser.add_argument('--crop_w', type=int, default=576)
    parser.add_argument('--resize_h', type=int, default=-1)
    parser.add_argument('--resize_w', type=int, default=-1)
    parser.add_argument('--render_scale', type=int, default=1, help='generate higher resolution images')
    parser.add_argument('--single', type=str, default=None)  # train on a single scene
    parser.add_argument('--precomputed_depth_path', type=str, default=None) # the depth map path

    '''model args'''
    parser.add_argument('--HR', type=int, default=False)
    parser.add_argument('--freeze_shader', type=int, default=False)
    parser.add_argument('--sphere_radius', type=float, default=7.5e-4)
    parser.add_argument('--free_xyz', type=int, default=False)
    parser.add_argument('--free_opy', type=int, default=False)
    parser.add_argument('--free_rad', type=int, default=False)
    parser.add_argument('--blend_gamma', type=float, default=1e-4, help='gamma for blending. See the Pulsar paper for details')
    parser.add_argument('--do_2d_shading', type=int, default=False)
    parser.add_argument('--shader_arch', type=str, default='simple_unet')
    parser.add_argument('--shader_norm', type=str, default='none', help='choice of normalization layers')
    parser.add_argument('--basis_type', type=str, default='mlp', help="the basis type to use for modeling the non-Lambertian effect. option: mlp;SH;none")
    parser.add_argument('--shader_output_channel', type=int, default=128)
    parser.add_argument('--pts_dropout_rate', type=float, default=0.0)
    parser.add_argument('--dim_pointfeat', type=int, default=16)
    parser.add_argument('--do_random_affine', type=int, default=False)
    parser.add_argument('--do_check_depth_consistency', type=int, default=True, help="do point pruning based on view consistency. default is True.")
    parser.add_argument('--max_num_pts', type=int, default=1000000000)
    parser.add_argument('--rasterize_rounds', type=int, default=5)
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir('pointclouds'):
        os.mkdir('pointclouds')

    train(args)





