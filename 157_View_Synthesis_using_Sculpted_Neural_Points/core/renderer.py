import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('core')

import os
import cv2
import subprocess
from collections import OrderedDict


autocast = torch.cuda.amp.autocast
# import matplotlib.pyplot as plt
import time

from raft import RAFT
from projector import ZbufferModelPts
from modules.extractor import BasicEncoder, SameResEncoder

EPS = 1e-2


def bilinear_sample2d(im, x, y):
    # x,y has shape B x N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int32).cuda() * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    return output

def grid_sample(image, ix, iy):
    # a differentiable version of grid_sample
    # copied and adapted from https://github.com/pytorch/pytorch/issues/34704#issuecomment-878940122
    N, C, IH, IW = image.shape
    _, H, W = ix.shape
    # _, H, W, _ = optical.shape

    # ix = optical[..., 0]
    # iy = optical[..., 1]
    #
    # ix = ((ix + 1) / 2) * (IW-1)
    # iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    # mask = (ix_nw >= 0).float() * (ix_nw <= IW - 1).float() * (iy_nw >= 0).float() * (iy_nw <= IH - 1).float()

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    # return out_val, mask
    return out_val

def batch_theta_matrix(poses):
    # poses are B x N x 4 x 4 matrices
    delta_pose = torch.matmul(poses[:, :, None], torch.inverse(poses[:, None, :]))
    dR = delta_pose[:, :, :, :3, :3] # B x N x N x 3 x 3
    cos_theta = (torch.einsum('...ii->...', dR) - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    return torch.rad2deg(torch.arccos(cos_theta)) # B x N x N

# class Renderer(nn.Module):
#     def __init__(self, **params):
#         super(Renderer, self).__init__()
#         self.raft = RAFT(**params)
#
#         HR = params["HR"]
#         factor = 8 if not HR else 4
#         h, w = params['crop_h'] // factor, params['crop_w'] // factor
#         self.feat_projector = ZbufferModelPts(output_size=(h, w), max_npoints=(params['num_frames']-1)*h*w)
#         self.params = params
#         self.detach_depth = params['detach_depth']
#         self.feature_extractor = PointFeatureExtractor(h, w)
#
#     def load_raft_param(self, restore_ckpt):
#         print('restoring raft parameters from %s' % restore_ckpt)
#         if restore_ckpt is not None:
#             tmp = torch.load(restore_ckpt)
#             if list(tmp.keys())[0][:7] == "module.":
#                 # self.raft = nn.DataParallel(self.raft)
#                 new_state_dict = OrderedDict()
#                 for k, v in tmp.items():
#                     name = k[7:]  # remove `module.`
#                     new_state_dict[name] = v
#                 # load params
#                 self.raft.load_state_dict(new_state_dict, strict=False)
#             # self.raft.load_state_dict(tmp, strict=False)
#             else:
#                 self.raft.load_state_dict(tmp, strict=False)
#
#     def forward(self, images, poses, intrinsics, depths, graph):
#         self.feature_extractor(images, poses, intrinsics, depths)
#         assert (self.params['output_appearance_features'])
#         eps = 1e-8
#         disp_est, apperance_features = self.raft(images, poses, intrinsics, graph)
#         depth = 1. / (disp_est[-1] + eps)
#         if self.detach_depth:
#             depth = depth.detach()
#         rgb_est = self.feat_projector(images, poses, intrinsics, depth, points_features=apperance_features)
#
#         return disp_est, rgb_est


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128, params={}):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        kernel_z = params['kernel_z']  # 3
        kernel_r = params['kernel_r']  # 3
        kernel_q = params['kernel_q']  # 3
        self.convz = nn.Conv2d(h_planes + i_planes, h_planes, kernel_z, padding=kernel_z // 2)
        self.convr = nn.Conv2d(h_planes + i_planes, h_planes, kernel_r, padding=kernel_r // 2)
        self.convq = nn.Conv2d(h_planes + i_planes, h_planes, kernel_q, padding=kernel_q // 2)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)
        z = torch.sigmoid(self.convz(net_inp))
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)))
        net = (1 - z) * net + z * q
        return net


class Renderer(nn.Module):
    def __init__(self, **params):
        super(Renderer, self).__init__()

        self.n_neighbors = params['fe_n_neighbors'] # 4
        self.dim_inp = dim_inp = params['fe_dim_inp'] # 128
        # self.dim_fmap = dim_fmap = params['fe_dim_fmap']  # 128
        self.dim_net = dim_net = params['fe_dim_net'] # 128
        dim0_delta = params['fe_dim0_delta']  # 256
        kernel0_delta = params['fe_kernel0_delta']  # 3
        kernel1_delta = params['fe_kernel1_delta']  # 3
        self.dim_pointfeat = params["fe_dim_pointfeat"]  # 256
        self.render_iters = params["fe_render_iters"] # 8
        self.output_opacity = params["fe_output_opacity"]  # false
        self.bkg_color = params["bkg_color"]  # false

        self.params = params

        HR = self.params["HR"]
        self.factor = 8 if not HR else 4
        H, W = params['crop_h'] // self.factor, params['crop_w'] // self.factor

        # H = params['out_h']
        # W = params['out_w']

        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1)  # B x 3 x N

        self.register_buffer("xyzs", xyzs)

        # self.neighbors_index_list = torch.tensor([[i for i in range(params['num_frames'] - 1) if i != j] for j in range(params['num_frames'] - 1)]).reshape(-1)

        # networks
        self.reference_encoder = BasicEncoder(output_dim=dim_net + dim_inp, norm_fn='instance', HR=params["HR"])
        # self.recon_encoder = BasicEncoder(output_dim=dim_inp, norm_fn='instance', HR=params["HR"])
        self.recon_encoder = SameResEncoder(output_dim=dim_inp, norm_fn='instance')

        self.gru = ConvGRU(h_planes=dim_net, i_planes=4*dim_inp, params=params)

        self.delta_decoder = nn.Sequential(
            nn.Conv2d(dim_net, dim0_delta, kernel0_delta, padding=kernel0_delta // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim0_delta, self.dim_pointfeat, kernel1_delta, padding=kernel1_delta // 2))

        render_scale = 1
        # render_scale = self.factor # as input res

        self.feat_projector = ZbufferModelPts(output_size=(H, W), feat_dim=self.dim_pointfeat, max_npoints=(params['num_frames'] - 1) * H * W, render_scale=render_scale, HR=params["HR"], rasterize_img=False, output_opacity=self.output_opacity, bkg_color=self.bkg_color, use_relative_viewdir=True) # render at the original resolution


    def forward(self, images, poses, intrinsics, depths_lowres):
        # images: B x N x 3 x H x W
        # poses: B x N x 4 x 4
        # intrinsics: B x N x 3 x 3
        # depths_lowres: B x N x 1 x h x w
        B, N, _, H, W = images.shape

        intrinsics_scaled = intrinsics.clone()

        intrinsics_scaled[:, :, 0] /= self.factor
        intrinsics_scaled[:, :, 1] /= self.factor

        h = H // self.factor
        w = W // self.factor

        images = images.clone()

        images *= 2 / 255.
        images -= 1

        _, input_images = torch.split(images, [1, N - 1], 1)
        _, input_depths = torch.split(depths_lowres, [1, N - 1], 1)

        with autocast(enabled=False):
            net_inp = self.reference_encoder(input_images.reshape(B*(N-1), 3, H, W))
            net_inp = net_inp.reshape(B, N-1, -1, h, w)

            net, feats = net_inp.split([self.dim_net, self.dim_inp], dim=2)

            feats_target_recon = None

            point_features = torch.zeros(B, N-1, self.dim_pointfeat, h, w)
            point_features = point_features.to(images.device).float()

            rendered_images = []

            for itr in range(self.render_iters):
                sampled_features_aggregated = self.multiview_feature_extraction(feats, poses, intrinsics_scaled, input_depths, feats_target_recon)
                # sampled_features_aggregated = feats
                # sampled_features_aggregated: B x N-1 x 4*dim_inp x h x w
                # feed into gru
                inp_shape = (B * (N-1), -1, h, w)
                out_shape = (B, N-1, -1, h, w)

                net_ = self.gru(net.view(*inp_shape), sampled_features_aggregated.view(*inp_shape))
                net = net_.view(*out_shape)

                delta = self.delta_decoder(net_).view(*out_shape) # B x N-1 x dim_pointfeat x h x w
                point_features += delta

                # # debug
                # point_features = feats

                # render w/ point features
                with autocast(enabled=False):
                    rgb_est = self.feat_projector(images, poses, intrinsics, input_depths, points_features=[point_features])[0] # expect a list here for pointfeatures. just create a dummy one
                # rgb_est has shape B x H x W x 3
                rgb_est = rgb_est.permute(0, 3, 1, 2) # B x 3 x H x W, range [-0.5, 0.5]

                # extract feature from this image
                # feats_target_recon = self.recon_encoder(rgb_est.detach()) # B x dim_inp x h x w
                feats_target_recon = self.recon_encoder(rgb_est)  # B x dim_inp x h x w

                rendered_images.append(rgb_est)

        return rendered_images

    def multiview_feature_extraction(self, feats, poses, intrinsics, depths, feats_target_recon=None):
        # feats: B x N-1 x C x h x w
        # poses: B x N x 4 x 4
        # intrinsics: B x N x 3 x 3
        # depths_lowres: B x N-1 x 1 x h x w

        B, _, C, h, w = feats.shape
        N = poses.shape[1]
        # intrinsics = intrinsics.clone()
        #
        # HR = self.params["HR"]
        # factor = 8 if not HR else 4
        # intrinsics[:, :, 0] /= factor
        # intrinsics[:, :, 1] /= factor

        # h = H // factor
        # w = W // factor

        # images = F.avg_pool2d(images.reshape(-1, C, H, W), factor).reshape(B, N, C, h, w) # debug! later replace with cnn

        # _, input_feat = torch.split(feats, [1, N - 1], 1)
        target_intrinsics, input_intrinsics = torch.split(intrinsics, [1, N - 1], 1)
        target_poses, input_poses = torch.split(poses, [1, N - 1], 1)
        # _, input_depths = torch.split(depths, [1, N - 1], 1)



        # for each target image, extract the nearby features + target features
        n_neighbors = min(self.n_neighbors, N-2) # all input views but itself
        # n_neighbors = N - 2 # all input views but itself

        # we want every tensor to have shape B*N*n_neighbors x ...
        depths_ref = depths.reshape(B, N-1, 1, 1, h, w).repeat(1, 1, n_neighbors, 1, 1, 1).reshape(-1, 1, h*w)

        K_cam1 = input_intrinsics.reshape(B, N-1, 1, 3, 3).repeat(1, 1, n_neighbors, 1, 1).reshape(-1, 3, 3)

        # for each camera, this should be the intrinsics of all its neighbors
        # inp_index_list = torch.arange(N-1).cuda().repeat_interleave(n_neighbors) # [0,0,0,...,0,1,1,1,...,1,...]
        neighbors_index_list = torch.tensor([np.random.choice(np.array([i for i in range(N-1) if i != j]), n_neighbors, replace=False) for j in range(N-1)]).to(feats.device).reshape(-1) # [1,2,3,...,N-1, 0,2,3,...]
        # neighbors_index_list = torch.tensor([[i for i in range(N - 1) if i != j] for j in range(N - 1)]).to(feats.device).reshape(-1)  # [1,2,3,...,N-1, 0,2,3,...]
        # neighbors_index_list = self.neighbors_index_list
        K_cam2 = input_intrinsics[:, neighbors_index_list, :, :].reshape(-1, 3, 3)

        RT_cam1 = input_poses.reshape(B, N-1, 1, 4, 4).repeat(1, 1, n_neighbors, 1, 1).reshape(-1, 4, 4)
        RT_cam2 = input_poses[:, neighbors_index_list, :, :].reshape(-1, 4, 4)

        sample_idx = self.project_pts(depths_ref, K_cam2, K_cam1, RT_cam1, RT_cam2) # B*N-1*n_neighbors x 2 x (h*w)
        # sample_idx = sample_idx.reshape(B, N-1, n_neighbors, 2, h, w) # this is the index we want to look into. the 2 dim stores the xy value.

        x = sample_idx[:, 0]
        y = sample_idx[:, 1]
        # x = sample_idx[:, 0:1]
        # y = sample_idx[:, 1:2]

        # after indexing, the returned matrix should have shape B x N-1 x n_neighbors x C x h x w
        feat_cam2 = feats[:, neighbors_index_list, :, :, :].reshape(-1, C, h, w).detach() # unknown bug here. detach for ad-hoc solution
        # perform the sampling
        sampled_features = bilinear_sample2d(feat_cam2, x, y) # B*N-1*n_neighbors x C x (h*w)
        # sampled_features = grid_sample(feat_cam2, x, y) # B*N-1*n_neighbors x C x 1 x (h*w)
        sampled_features = sampled_features.reshape(B, N-1, n_neighbors, C, h, w)

        # # do vis here
        # source_to_vis = input_img[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # cv2.imwrite("/u/yz7608/view-syn/temp/source_vis_%08d.png" % 0, source_to_vis)
        # for i in range(n_neighbors):
        #     target_to_vis = sampled_features[0, 0, i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        #     cv2.imwrite("/u/yz7608/view-syn/temp/target_vis_%08d_view%08d.png" % (0, i), target_to_vis)
        #
        # time.sleep(1.0)
        # assert(False)

        # aggregate the neighbors
        sampled_features_mean = sampled_features.mean(dim=2)
        sampled_features_var = sampled_features.std(dim=2)

        # sampled_features_aggregated = torch.cat([feats, sampled_features_mean, sampled_features_var],
        #                                          dim=2)

        # now aggregate information from the target_recon
        if feats_target_recon is None:
            sampled_recon = torch.zeros_like(feats)
        else:
            depths_ref = depths.reshape(-1, 1, h*w) # B*N-1 x 1 x (h*w)
            K_cam1 = input_intrinsics.reshape(-1, 3, 3)
            K_cam2 = target_intrinsics.repeat(1, N-1, 1, 1).view(-1, 3, 3)

            RT_cam1 = input_poses.reshape(-1, 4, 4)
            RT_cam2 = target_poses.repeat(1, N-1, 1, 1).view(-1, 4, 4)

            sample_idx = self.project_pts(depths_ref, K_cam2, K_cam1, RT_cam1, RT_cam2) # B*N-1 x 2 x (h*w)

            x = sample_idx[:, 0]
            y = sample_idx[:, 1]

            # todo: extract feature from it
            # feat_recon = feats_target_recon
            feats_target_recon = feats_target_recon.repeat(1, N-1, 1, 1, 1).reshape(B*(N-1), -1, h, w)

            sampled_recon = bilinear_sample2d(feats_target_recon, x, y)  # B*N-1*n_neighbors x C x (h*w)
            sampled_recon = sampled_recon.reshape(B, N - 1, -1, h, w)

        # sampled_features_aggregated = torch.cat([feats, sampled_features_mean, sampled_recon],
        #                                         dim=2)  # B x N-1 x 4C x h x w
        sampled_features_aggregated = torch.cat([feats, sampled_features_mean, sampled_features_var, sampled_recon],
                                                dim=2)  # B x N-1 x 4C x h x w
        # sampled_features_aggregated = torch.cat([feats, sampled_recon],
        #                                         dim=2)  # B x N-1 x 4C x h x w

        return sampled_features_aggregated


    def project_pts(
        self, depth, K_cam2, K_cam1, RT_cam1, RT_cam2
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        # K and K_inv should have shape B x 3 x 3
        # depth has shape B x 1 x N, storing the z values (positive pointing to the obj)
        Kinv_cam1 = torch.inverse(K_cam1)
        RTinv_cam1 = torch.inverse(RT_cam1)

        projected_coors = self.xyzs * depth # B x 3 x N

        # Transform into camera coordinate of the first view
        cam1_X = Kinv_cam1.bmm(projected_coors) # B x 3 x N, xyz in cam1 space
        cam1_X = torch.cat((cam1_X, torch.ones_like(cam1_X[:, 0:1])), dim=1) # B x 4 x N, turned into homogeneous coord

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        cam2_X = RT.bmm(cam1_X)
        cam2_X = cam2_X[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        # And intrinsics
        xy_proj = K_cam2.bmm(cam2_X) # B x 3 x N

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        xy_proj = xy_proj[:, 0:2, :] / zs # B x 2 x N

        return xy_proj








