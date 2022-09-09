import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from models.losses.multi_view_depth_loss import multi_view_depth_loss
from models.losses.synthesis import SynthesisLoss
from models.networks.depth_regressor import get_depth_regressor
from models.networks.decoder import Decoder
from models.networks.coding import PositionalEncoding
from models.networks.utilities import get_encoder
from models.networks.transformer import Image_Fusion_Transformer
from models.projection.z_buffer_manipulator import get_ptsmanipulator
from models.depth_es.depth_estimator import Patch_Depth_ES

class Multi_Z_Transformer(nn.Module):
    """
    FWD model which takes multiple posed inputs and outputs novel view images.
    In this model, we assume the input is taken as BS, num_in, C, H, W and outout is BS, C, H, W.
    """
    def __init__(self, opt):
        super().__init__()

        ##### LOAD PARAMETERS
        opt.decode_in_dim = 1
        self.opt = opt
        # Use H if specifid in opt or H = W
        if not hasattr(self.opt, "H"):
            self.H = opt.W
        else:
            self.H = opt.H
        self.W = opt.W
        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
            "max_z", torch.Tensor([self.opt.max_z])
        )
        self.cam_coord = opt.cam_coord
        self.scale_factor = opt.scale_factor
        self.inverse_depth_com = self.opt.inverse_depth_com
        self.project_out_scale = opt.project_out_rescale
        activation_func = nn.ReLU(inplace=True)

        #### REGRESSION DEPTHS AND ENCODING FEATURES.
        # Point cloud depth regressor
        self.pts_regressor = get_depth_regressor(opt)
        # Wether using depth completions. It is expected when taking sensor depths or MVS estimated depths as inputs.
        self.depth_com = opt.depth_com 
        # Whether use MVS to estimate depths
        if opt.mvs_depth:
            self.mvs_depth_estimator = Patch_Depth_ES(opt)

        # Encode features to a given resolution.
        # If use RGB as features, we don't apply encoder to input images.
        # Otherwise, we apply encoder to the images and get 64-dim feature maps. 
        if not self.opt.use_rgb_features:
            self.encoder = get_encoder(opt)
            out_dim = 64
        else:
            out_dim = 3
    
        # View dependent feature MLP
        if opt.view_dependence:
            if self.opt.use_rgb_features:
                self.vd_1 = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, 4))
                self.vd_2 = nn.Sequential(nn.Linear(7, 16),
                                        activation_func,
                                        nn.Linear(16,3))
            else:
                self.vd_1 = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16,32))
                in_dim = 32 + 64
                out_dim = 64
                self.vd_2 = nn.Sequential(nn.Linear(in_dim, 64),
                                        activation_func,
                                        nn.Linear(64,out_dim))
        
        ##### POINT CLOUD MANIPULATIONS AND RENDERER.
        PtsManipulator = get_ptsmanipulator(opt.cam_coord)
        if self.opt.down_sample:
            if self.opt.use_rgb_features:
                self.pts_transformer = PtsManipulator(opt.W//2, H=opt.H//2, C=3, opt=opt)
            else:
                self.pts_transformer = PtsManipulator(opt.W//2, H=opt.H//2, opt=opt)
        else:
            if self.opt.use_rgb_features:
                self.pts_transformer = PtsManipulator(opt.W, H=opt.H, C=3, opt=opt)
            else:
                self.pts_transformer = PtsManipulator(opt.W, H=opt.H, opt=opt)

        # Calculate the input dim for generator
        if self.opt.use_rgb_features:
            decode_in_dim = 3
        else:
            decode_in_dim = 64

        # Whether or not using geometric information (depths, view direction changes) as position encoding of transformers.
        if opt.geo_position_encoding:
            self.ps = nn.Sequential(nn.Linear(5, 32),
                                    activation_func,
                                    nn.Linear(32, out_dim))
        
        # Whether add extra geo feature to points. We don't use it in the paper.
        if self.opt.render_geo:
            # if apply coordinate encoding
            if self.opt.geo_encoding:

                if self.opt.geo_type == "z":
                    d_in = 1
                elif self.opt.geo_type == 'xyz':
                    d_in = 3
                self.geo_encoder = PositionalEncoding(num_freqs=self.opt.encoding_num_freqs, d_in=d_in, freq_factor=np.pi, include_input=self.opt.encoding_include_input)
                decode_in_dim += self.geo_encoder.d_out
            # if only pad z coordinate
            else:
                if self.opt.geo_type == 'z':
                    decode_in_dim += 1

                elif self.opt.geo_type == 'xyz':
                    decode_in_dim += 3

        ##### FUSION AND DECODER.
        # if not use transformer, we stack the projected image as input for generator.
        if not self.opt.use_transformer:
            decode_in_dim = decode_in_dim * opt.input_view_num
        opt.decode_in_dim = decode_in_dim
        self.opt.decode_in_dim = opt.decode_in_dim
        self.fusion_module = Image_Fusion_Transformer(opt)
        self.decoder = Decoder(opt, norm=opt.decoder_norm)

        ##### LOSS FUNCTION
        # Module to abstract away the loss function complexity
        self.loss_function = SynthesisLoss(opt=opt)
        # whether using multiview consistent loss.
        if self.opt.consis_loss:
            self.consis_loss = multi_view_depth_loss(opt)

    def data_process(self, batch):
        """
        Prepare the input batch data.
        """
        input_imgs = []
        output_imgs = []
        input_masks = []
        output_masks = []
        num_inputs = self.opt.input_view_num
        num_outputs = len(batch['images']) - num_inputs

        for i in range(num_inputs):
            input_img = deepcopy(batch["images"][i])
            input_mask = deepcopy(batch["masks"][i])
            H, W = input_img.shape[-2:]
            if  self.opt.down_sample:
                input_img = F.interpolate(input_img, size=(H//2, W//2), mode="area")
            input_imgs.append(input_img)
            input_masks.append(input_mask)
        input_imgs = torch.stack(input_imgs, 1) # B x num_inputs x C x H x W
        input_masks = torch.stack(input_masks, 1) # B x num_inputs x 1 x H x W
        for i in range(num_outputs):
            output_imgs.append(batch["images"][i+num_inputs])
            output_masks.append(batch['masks'][i+num_inputs])
        output_imgs = torch.stack(output_imgs, 1) # B x num_outputs x C x H x W

        output_masks = torch.stack(output_masks, 1) # B x num_outputs x 1 x H x W

        # Camera parameters
        K = deepcopy(batch["cameras"][0]["K"]).clone()
        K_inv = deepcopy(batch["cameras"][0]["Kinv"]).clone()
        if self.opt.down_sample:
            K[:, 0:2, 0:3] = K[:, 0:2, 0:3] / 2.0
            K_inv = torch.inverse(K)

        input_RTs = []
        input_RTinvs = []
        for i in range(num_inputs):
            input_RTs.append(batch["cameras"][i]["P"])
            input_RTinvs.append(batch["cameras"][i]["Pinv"])
        input_RTs = torch.stack(input_RTs, 1)
        input_RTinvs = torch.stack(input_RTinvs, 1)
        output_RTs =  []
        output_RTinvs = []
        for i in range(num_outputs):
            output_RTs.append(batch["cameras"][i+num_inputs]["P"])
            output_RTinvs.append(batch["cameras"][i+num_inputs]["Pinv"])
        output_RTs = torch.stack(output_RTs, 1)
        output_RTinvs = torch.stack(output_RTinvs, 1)

        if torch.cuda.is_available():
            input_imgs = input_imgs.cuda()
            output_imgs = output_imgs.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            input_RTs = input_RTs.cuda()
            input_RTinvs = input_RTinvs.cuda()

            output_RTs = output_RTs.cuda()
            output_RTinvs = output_RTinvs.cuda()
            input_masks = input_masks.cuda()
            output_masks = output_masks.cuda()

        return input_imgs, output_imgs, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs, input_masks, output_masks

    def get_init_depth(self, batch):
        """
        Get init depth.
        We could get depth from provided sensor depths or return None or using MVS algorithm to get deptns.
        """
        num_inputs = self.opt.input_view_num
        gt_depth = None
        mvs_depth = None

        # Use provided incomplete sensor depths.
        if self.opt.use_gt_depth:
            depth_imgs = []
            if "depths" in batch.keys():
                for i in range(num_inputs):
                    if torch.cuda.is_available():
                        depth_imgs.append(batch["depths"][i].cuda())
                    else:
                        depth_imgs.append(batch["depths"][i])
            if self.opt.down_sample:
                depth_imgs = torch.cat(depth_imgs, 1)
                gt_depth = F.interpolate(depth_imgs, size=(self.H//2, self.W//2), mode="nearest").unsqueeze(2)
            else:
                gt_depth = torch.stack(depth_imgs, 1) # B x num_outputs x 1 x H x W
    
        # Using MVS module to get depths.
        if self.opt.mvs_depth and self.opt.depth_com:
            # Fully differentiable MVS module
            if self.opt.learnable_mvs:
                results, output, condfidence= self.mvs_depth_estimator(batch)
                mvs_depth = torch.stack(results, 1)
            else:
                with torch.no_grad():
                    results, output, condfidence = self.mvs_depth_estimator(batch)
                mvs_depth = torch.stack(results,1)
        return gt_depth, mvs_depth

    def depth_com_module(self, depth_img, input_img, input_RTs, K, batch=None):
        """
        Depth Completion module.

        If the depth_img from get_init_depth is None, we need to estimate the depths.
        If the depth_img is not None, we complete the depths.
        """
        opacity = None
        # If we have incomplete depth for completion
        if self.opt.depth_com:
            # Using inverse depths.
            if self.inverse_depth_com:
                inverse_depth_img = 1. / torch.clamp(depth_img, min=0.001)
                inverse_depth_img[depth_img < 0.001] = 0
                ref_depth = inverse_depth_img.detach()
                # Normalize the depth
                if self.opt.normalize_depth:
                    inverse_depth_img = inverse_depth_img / ( 1.0 / self.opt.min_z)
            else:
                ref_depth = depth_img.detach()
            
            # Concat input_img and depth_img together for regressor.
            if self.inverse_depth_com:
                depth_input = torch.cat((input_img, inverse_depth_img), dim=1)
            else:
                depth_input = torch.cat((input_img, depth_img), dim=1)
            regressed_pts, opacity, refine_depth = self.pts_regressor(depth_input, input_RTs, K)
            return regressed_pts, opacity, ref_depth, refine_depth
        else:
            # Directly regressing depths from images.
            regressed_pts, opacity, refine_depth = self.pts_regressor(input_img, input_RTs, K)
            return regressed_pts, opacity, None, refine_depth

    def add_geo_feature(self, point_feature, point_pos):
        """
        Based on the point positions, we concentrate extra geometric feature to point features.
        We don't do it in our model.
        Inputs:
            -- point_pos: Batch x 3 x N. Input point positions.
            -- point_feature: Batch x c x N. Input point features.
        We don't use it for paper.
        """
        # Whether we add geometric information into rendered features.
        if self.opt.render_geo:
            # Whether apply positional encoding to geometric information.
            if self.opt.geo_encoding:
                bs, _, N = point_pos.shape
                # Only insert depth information.
                if self.opt.geo_type == "z":
                    geo_feature = point_pos[:,2:3,:].permute(0,2,1).view(-1, 1)
                    geo_feature = ( 1./geo_feature ) / ( 1.0 / self.opt.min_z)
                    geo_feature = geo_feature * 2.0 - 1.0
                # Insert x, y, z information.
                elif self.opt.geo_type == 'xyz':
                    geo_feature = point_pos[:, :, :]
                    geo_feature[:,2,:] = (1.0 / geo_feature[:,2,:]) / ( 1.0 / self.opt.min_z) * 2.0 - 1.0
                    geo_feature = geo_feature.permute(0, 2, 1).view(-1, 2)
                geo_feature = self.geo_encoder(geo_feature)
                geo_feature = geo_feature.view(bs, N, -1).contiguous().permute(0, 2, 1)
            else:
                # Not apply positional encoding to geometric inforamtion.
                if self.opt.geo_type == 'z':
                    # Insert inverse depth information.
                    geo_feature = point_pos[:, 2:3, :]
                    geo_feature = ( 1./geo_feature ) / ( 1.0 / self.opt.min_z)
                    geo_feature = geo_feature * 2.0 - 1.0
                elif self.opt.geo_type == 'xyz':
                    geo_feature = point_pos[:, :, :]
                    geo_feature[:,2,:] = (1.0 / geo_feature[:,2,:]) / ( 1.0 / self.opt.min_z) * 2.0 - 1.0
            return torch.cat((point_feature, geo_feature), dim=1)
        else:
            return point_feature
 
    def compute_view_dir_change(self, xyz, source_camera, target_camera):
        '''
        Compute the view direction change from source camera to target camera. 
        Inputs:
            -- xyz: [BS, N, 3], points positions. BS: batch size. N: point numbers. 
            -- source_camera: [BS, 4, 4]
            -- target_camera: [BS, nviews, 4, 4]
        Outputs:
            -- [BS, num_views, N, 4]; The first 3 channels are unit-length vector of the difference between
            query and target ray directions, the last channel is the inner product of the two directions.
        '''
        BS, N, _ = xyz.shape
        ray2tar_pose = (source_camera[:, :3, 3].unsqueeze(1) - xyz).unsqueeze(1) # Bs x 1 x N x 3
        ray2tar_pose = ray2tar_pose /(torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (target_camera[:, :, :3, 3].unsqueeze(2) - xyz.unsqueeze(1)) # Bs x nviews x N x 3
        ray2train_pose = ray2train_pose / (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        return ray_diff # Bs x nviews x N x C
    
    def compute_view_dir_change_unbatched(self, xyz, source_camera, target_camera):
        '''
        Compute the view direction change from source camera to target camera. 
        Inputs:
            -- xyz: [BS, N, 3], points positions. BS: batch size. N: point numbers. 
            -- source_camera: [BS, 4, 4]
            -- target_camera: [BS, 4, 4]
        Outputs:
            -- [BS, N, 4]; The first 3 channels are unit-length vector of the difference between
            query and target ray directions, the last channel is the inner product of the two directions.
        '''
        BS, N, _ = xyz.shape
        ray2tar_pose = (source_camera[:, :3, 3].unsqueeze(1) - xyz)
        ray2tar_pose = ray2tar_pose /(torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (target_camera[:, :3, 3].unsqueeze(1) - xyz)
        ray2train_pose = ray2train_pose / (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        return ray_diff

    def view_render(self, src, pred_pts, opacity, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs):
        """
        We construct point cloud for each input and render them to target views individually.
        Inputs: 
            -- src: shape: BS x num_inputs x C x N. Input points features.
            -- pred_pts: BS x num_inputs x 1 x N. Input points depths.
            -- opacity: None or BS x num_inputs x 1 x N. Input points opacities or None.
            -- K: BS x 4 x 4. Intrinsic matrix. 
            -- input_RTs: BS x num_inputs x 4 x 4. Input camera matrixes. 
            -- output_RTs: BS x num_outputs x 4 x 4. Target camera matrixes. 
        Outputs:
            -- rendered_images
            -- rendered_depths or None
        """
        num_inputs = self.opt.input_view_num
        num_outputs = output_RTs.shape[1]
        results = []
        rendered_depths = []
        bs, nv, C, N = src.shape 
        for i in range(num_inputs):
            # From input camera coordinate to world coordinate.
            pts_3D_nv = self.pts_transformer.view_to_world_coord(pred_pts[:, i], K, K_inv, input_RTs[:, i], input_RTinvs[:, i])
            src_nv = src[:, i:i+1]
            if opacity is not None:
                opacity_nv = opacity[:, i:i+1, 0]
                opacity_nv = opacity_nv.expand(-1, num_outputs, -1).view(-1, 1, N)
            else:
                opacity_nv = None

            # Whether apply view-dependent feature MLP or not.
            if self.opt.view_dependence:
                ray_diff = self.compute_view_dir_change(pts_3D_nv[:,0:3,].permute(0,2,1), input_RTinvs[:, i], output_RTinvs)
                modified_src = self.vd_1(ray_diff)# Bs x nviews x N x C
                modified_src = torch.cat((src_nv.expand(-1, num_outputs, -1, -1).permute(0, 1, 3, 2), modified_src), dim=-1)
                modified_src = self.vd_2(modified_src).permute(0, 1, 3, 2)
                sampler = self.pts_transformer.world_to_view(pts_3D_nv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, N), K.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4), K_inv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4), output_RTs.view(-1, 4, 4), output_RTinvs.view(-1, 4,4))
                modified_src = self.add_geo_feature(modified_src, sampler)
                pointcloud = sampler.permute(0, 2, 1).contiguous()# bs*n_output x N x 1
                modified_src = modified_src.view(-1, *modified_src.shape[2:]) #bs*n_output x N x C
            else:
                sampler = self.pts_transformer.world_to_view(pts_3D_nv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, N), K.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4), K_inv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4), output_RTs.view(-1, 4, 4), output_RTinvs.view(-1, 4,4))
                modified_src = self.add_geo_feature(src_nv.expand(-1, num_outputs, -1, -1), sampler)
                pointcloud = sampler.permute(0, 2, 1).contiguous()
                modified_src = modified_src.view(-1, *modified_src.shape[2:]) #bs*n_output x N x C

            # We also render depths as geometric features for position encoding in Transformer. 
            if self.opt.geo_position_encoding:
                result, rendered_depth = self.pts_transformer.splatter(pointcloud, modified_src, opacity_nv, depth=True)
                results.append(result)
                rendered_depths.append(rendered_depth)
            else:
                results.append(self.pts_transformer.splatter(pointcloud, modified_src))
        
        if self.opt.geo_position_encoding:
            return torch.stack(results, 0), torch.stack(rendered_depths, 0) # num_inputs, bs x num_outputs, C, H, W
        else:
            return torch.stack(results, 0), None

    def pos_end(self, gen_depth, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs):
        """
        Positional encoding for fusion transformer. 
        Given the rendered depth of output view image pixels, we calculate the position encoding of each pixel for fusion transformer.
        Inputs:
            -- gen_depths: num_inputs, BS*num_outputs, _, H, W. The rendered depths for each point.
            -- K: BS x 4 x 4. Intrinsic matrixes.
            -- K_inv: Inverse Intrinsic matrixes.
            -- input_RTs: BS x num_inputs x 4 x 4. Input camera matrixes.
            -- input_RTinvs: BS x num_inputs x 4 x 4. Inverse input camera matrixes.
            -- output_RTs: BS x num_outputs x 4 x 4. Output camera matrixes.
            -- output_RTinvs: BS x num_outputs x 4 x 4. Inversed Output camera matrixes.
        Outputs:
            -- ray_feature: num_inputs x bs*num_outputs x 64 x H x W.
        """
        H, W = gen_depth.shape[-2], gen_depth.shape[-1]
        num_inputs = input_RTs.shape[1]
        num_outputs = output_RTs.shape[1]
        num_in, num_out, _, H, W = gen_depth.shape
        bs = K.shape[0]
        iRT = input_RTs.permute(1, 0, 2, 3).contiguous().unsqueeze(2).repeat(1,1,num_outputs, 1, 1)
        oRT = output_RTs.unsqueeze(0).repeat(num_inputs, 1, 1, 1, 1)
        K = K.unsqueeze(1).unsqueeze(0).repeat(num_inputs,1, num_outputs, 1, 1)
        
        K_viewed = K.view(num_inputs, -1, 4, 4).contiguous().view(-1, 4, 4)
        K_inv_viewed = torch.inverse(K_viewed)
        input_RTs = iRT.view(-1, *iRT.shape[3:])
        input_RTinvs = torch.inverse(input_RTs)
        output_RTs = oRT.view(-1, *oRT.shape[3:])
        output_RTinvs = torch.inverse(output_RTs)

        gen_depth_viewed = gen_depth.view(-1, *gen_depth.shape[2:])
        pts3D = self.pts_transformer.view_to_world_coord(gen_depth_viewed, K_viewed, K_inv_viewed, output_RTs, output_RTinvs) # num_inputs * bs * num_outputs x 4 x N
        
        ray_diff = self.compute_view_dir_change_unbatched(pts3D[:,0:3].permute(0, 2, 1), output_RTinvs, input_RTinvs) # num_inputs * bs * num_outputs x N x 4
        
        gen_depth_viewed = (gen_depth_viewed - self.opt.min_z) / (self.opt.max_z -self.opt.min_z)
        ray_diff = torch.cat((ray_diff, gen_depth_viewed.view(gen_depth_viewed.shape[0],1,-1).permute(0, 2, 1)), dim=-1)

        ray_feature = self.ps(ray_diff) # num_inputs * bs * num_outputs x H * W x 5

        ray_feature = ray_feature.permute(0, 2, 1).view(num_inputs, bs * num_outputs, ray_feature.shape[-1], H, W )

        return ray_feature

    def forward(self, batch):
        """
        Forward pass of a view synthesis model.
        """
        depth_loss = 0
        consis_loss = 0

        with profiler.record_function("forward"):
            with profiler.record_function("data_init"):
                num_inputs = self.opt.input_view_num
                num_outputs = len(batch['images']) - num_inputs
                gt_depth, mvs_depth = self.get_init_depth(batch) # list of depths  with the shape BxHxW.
                input_imgs, output_imgs, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs, _, _ = self.data_process(batch)

            with profiler.record_function("depth_regression"):
                bs, nv, C, H, W = input_imgs.shape
                input_imgs = input_imgs.view(-1, C, H, W) # BS*num_input, C, H, W
                if gt_depth is not None:
                    gt_depth = gt_depth.contiguous().view(-1, 1, H, W)
                if mvs_depth is not None:
                    mvs_depth = mvs_depth.contiguous().view(-1,1,H, W)
                if self.opt.use_rgb_features:
                    fs = input_imgs * 0.5 + 0.5
                else:
                    fs = self.encoder(input_imgs)
                    if self.opt.append_RGB:
                        fs = torch.cat([fs, input_imgs * 0.5 + 0.5], axis=1)
                        fs = fs.view(bs, nv, -1, H, W)
                if  self.opt.mvs_depth:
                    regressed_pts, opacity, ref_depth, refine_depth = self.depth_com_module(mvs_depth, input_imgs, input_RTs, K, batch)
                else:
                    regressed_pts, opacity, ref_depth, refine_depth = self.depth_com_module(gt_depth, input_imgs, input_RTs, K, batch)
                
                # Depth supervision loss.
                if self.opt.train_depth:
                    with torch.no_grad():
                        valid_depth_mask = gt_depth > 0.0
                    if self.opt.gt_depth_loss_cal is "normal":
                        depth_loss += nn.L1Loss()(regressed_pts[valid_depth_mask], gt_depth[valid_depth_mask]) * self.opt.gt_depth_loss_weight
                    elif self.opt.gt_depth_loss_cal is "inverse":
                        depth_loss += nn.L1Loss()(1.0/regressed_pts[valid_depth_mask], 1.0/gt_depth[valid_depth_mask]) * self.opt.gt_depth_loss_weight
                if self.opt.consis_loss:
                    consis_loss = self.consis_loss(regressed_pts, K, input_RTs)
                    
            with profiler.record_function("rendering"):
                bs, nv, c, _, _ = fs.shape
                fs = fs.view(bs, nv, c, -1).contiguous()
                regressed_pts = regressed_pts.contiguous().view(bs, nv, 1, -1)
                if opacity is not None:
                    opacity = opacity.contiguous().view(bs, nv, 1, -1) 

                gen_fs, gen_depth = self.view_render(fs, regressed_pts, opacity, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs)
                if self.opt.use_rgb_features:
                    gen_fs = gen_fs * 2.0 - 1.0

            # Whether use fusion transformer or casacaded generator.
            if self.opt.use_transformer:
                with profiler.record_function("transformer fusion"):
                    if self.opt.geo_position_encoding:
                        pos_enc = self.pos_end(gen_depth, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs)
                        gen_fs = self.fusion_module(gen_fs, pos_enc)
                    else:
                        gen_fs = self.fusion_module(gen_fs)
            else:
                num_in, num_out, C, H, W = gen_fs.shape
                gen_fs = gen_fs.permute(1, 0, 2, 3, 4).contiguous().view(num_out, -1, H, W)
            
            # Decoder & refinement.
            with profiler.record_function("generator"):
                gen_img = self.decoder(gen_fs)
                loss = self.loss_function(gen_img, output_imgs.view(-1, *output_imgs.shape[2:]))
            
            if self.opt.train_depth:
                loss["Total Loss"] += depth_loss
                loss["depth_loss"] = depth_loss
            if self.opt.consis_loss:
                loss["Total Loss"] += consis_loss
                loss["consis_loss"] = consis_loss
            if self.depth_com and not self.opt.learnable_mvs:
                return (
                    loss,
                    {
                        "InputImg": input_imgs,
                        "OutputImg": output_imgs.view(-1, *output_imgs.shape[2:]),
                        "PredImg": gen_img,
                        "PredDepth": ref_depth,
                        "ComDepth": refine_depth,
                        "ProjectedImg": gen_fs[:, :3]
                    },
                )
            else:
                return (
                    loss,
                    {
                        "InputImg": input_imgs,
                        "OutputImg": output_imgs.view(-1, *output_imgs.shape[2:]),
                        "PredImg": gen_img,
                        "PredDepth": refine_depth,
                        "ProjectedImg": gen_fs[:, :3]
                    },
                )

    def eval_one_step(self, batch, **kwargs):

        with profiler.record_function("forward"):
            with profiler.record_function("data_init"):
                num_inputs = self.opt.input_view_num
                num_outputs = len(batch['images']) - num_inputs
                gt_depth, mvs_depth = self.get_init_depth(batch) # list of depths  with the shape BxHxW.
                input_imgs, output_imgs, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs, _, _ = self.data_process(batch)

            with profiler.record_function("depth_regression"):
                bs, nv, C, H, W = input_imgs.shape
                input_imgs = input_imgs.view(-1, C, H, W) # BS*num_input, C, H, W
                if gt_depth is not None:
                    gt_depth = gt_depth.contiguous().view(-1, 1, H, W)
                if mvs_depth is not None:
                    mvs_depth = mvs_depth.contiguous().view(-1,1,H, W)
                if self.opt.use_rgb_features:
                    fs = input_imgs * 0.5 + 0.5
                else:
                    fs = self.encoder(input_imgs)
                    if self.opt.append_RGB:
                        fs = torch.cat([fs, input_imgs * 0.5 + 0.5], axis=1)
                        fs = fs.view(bs, nv, -1, H, W)
                if  self.opt.mvs_depth:
                    regressed_pts, opacity, ref_depth, refine_depth = self.depth_com_module(mvs_depth, input_imgs, input_RTs, K, batch)
                else:
                    regressed_pts, opacity, ref_depth, refine_depth = self.depth_com_module(gt_depth, input_imgs, input_RTs, K, batch)
            # breakpoint()
            with profiler.record_function("rendering"):
                bs, nv, c, _, _ = fs.shape
                fs = fs.view(bs, nv, c, -1).contiguous()
                regressed_pts = regressed_pts.contiguous().view(bs, nv, 1, -1)
                if opacity is not None:
                    opacity = opacity.contiguous().view(bs, nv, 1, -1) 

                gen_fs, gen_depth = self.view_render(fs, regressed_pts, opacity, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs)
                if self.opt.use_rgb_features:
                    gen_fs = gen_fs * 2.0 - 1.0

            # Whether use fusion transformer or casacaded generator.
            if self.opt.use_transformer:
                with profiler.record_function("transformer fusion"):
                    if self.opt.geo_position_encoding:
                        pos_enc = self.pos_end(gen_depth, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs)
                        gen_fs = self.fusion_module(gen_fs, pos_enc)
                    else:
                        gen_fs = self.fusion_module(gen_fs)
            else:
                num_in, num_out, C, H, W = gen_fs.shape
                gen_fs = gen_fs.permute(1, 0, 2, 3, 4).contiguous().view(num_out, -1, H, W)
            
            # Decoder & refinement.
            with profiler.record_function("generator"):
                gen_img = self.decoder(gen_fs)
            
            if self.depth_com and not self.opt.learnable_mvs:
                return {
                        "InputImg": input_imgs,
                        "OutputImg": output_imgs.view(-1, *output_imgs.shape[2:]),
                        "PredImg": gen_img,
                        "PredDepth": refine_depth,
                        "ComDepth": refine_depth,
                        "ProjectedImg": gen_fs[:, :3]
                    }
            else:
                return {
                        "InputImg": input_imgs,
                        "OutputImg": output_imgs.view(-1, *output_imgs.shape[2:]),
                        "PredImg": gen_img,
                        "PredDepth": refine_depth,
                        "ProjectedImg": gen_fs[:, :3]
                    }

    def eval_batch(self, batch, chunk=8, **kwargs):
        """
        Evaluation code. 
        We reorganize the data for evaluation by spliting data into several chunks.
        """
        num_inputs = kwargs.get('num_view', self.opt.input_view_num)
        num_outputs = len(batch["images"]) - num_inputs
        num_chunks = math.ceil( num_outputs / chunk)
        results = {}
        for chunk_idx in range(num_chunks):
            endoff = min(num_inputs + chunk_idx * chunk + chunk, num_outputs + num_inputs)
            start = num_inputs + chunk_idx * chunk
            instance_num = int(endoff - start)
            new_batch = {}
            new_batch['path'] = batch['path']
            new_batch['img_id'] = batch['img_id']
            new_batch['images'] = batch['images'][0:num_inputs]
            new_batch['images'] = [new_batch['images'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
            new_batch['images'].append(torch.cat(batch['images'][start:endoff], 0))

            new_batch['masks'] = batch['masks'][0:num_inputs]
            new_batch['masks'] = [new_batch['masks'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
            new_batch['masks'].append(torch.cat(batch['masks'][start:endoff],0))
            new_camera = {}
            for item in batch['cameras'][0]:
                new_camera[item] = []
            for instance in batch['cameras']:
                for item in instance:
                    new_camera[item].append(instance[item])

            camera_list = []
            for i in range(num_inputs):
                camera_tmp = {}
                for item in new_camera:
                    camera_tmp[item] = new_camera[item][i]
                    the_shape = camera_tmp[item].shape
                    new_shhape = (instance_num,) + the_shape[1:]
                    camera_tmp[item] = camera_tmp[item].expand(new_shhape).clone()
                camera_list.append(camera_tmp)
            camera_tmp = {}
            for item in new_camera:
                camera_tmp[item] = torch.cat(new_camera[item][start:endoff])
            camera_list.append(camera_tmp)
            new_batch['cameras'] = camera_list
            if "depths" in batch.keys():
                new_batch['depths'] = batch['depths'][0:num_inputs]
                new_batch['depths'] = [new_batch['depths'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
                new_batch['depths'].append(torch.cat(batch['depths'][start:endoff],0))
            result = self.eval_one_step(new_batch)
            results.update({chunk_idx:result})
        new_results = {}
        for term in results[0].keys():
            buffer = []
            for i in range(len(results)):
                buffer.append(results[i][term])
            new_results.update({term:torch.cat(buffer, 0)})
        return [None, new_results]
