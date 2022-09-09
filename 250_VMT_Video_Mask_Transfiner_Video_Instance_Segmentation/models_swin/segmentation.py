"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import random
import math

from kornia.morphology import dilation
import copy
import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from .position_encoding import PositionalEncoding3D

def pos_embed(x, temperature=10000, scale=2 * math.pi, normalize=True):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work on
    images.
    """
    batch_size, channel, height, width = x.size()
    mask = x.new_ones((batch_size, height, width))
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = channel // 2
    assert num_pos_feats * 2 == channel, (
        'The input channel number must be an even number.')
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

class VMT(nn.Module):
    def __init__(self, detr, rel_coord=True, freeze_detr=False):
        super().__init__()
        self.detr = detr
        self.rel_coord = rel_coord
        self.object_limit = 300
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
     
        self.in_channels = hidden_dim // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.max_insts_num = 100
        self.mask_out_stride = 4

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        weight_nums_inc = []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                    weight_nums_inc.append((self.in_channels + 2 + 1) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                weight_nums_inc.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                weight_nums_inc.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.weight_nums_inc = weight_nums_inc
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.num_gen_params_inc = sum(weight_nums_inc) + sum(bias_nums)
        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)
        
        self.mask_head = MaskHeadSmallConv(hidden_dim, None, hidden_dim)
        
        self.controller_inc = MLP(hidden_dim, hidden_dim, self.num_gen_params_inc, 3)
        for contr in self.controller_inc.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)
        
        self.controller_refine = MLP(hidden_dim, hidden_dim, self.num_gen_params_inc, 3)
        for contr in self.controller_refine.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        self.lay_hr1 = torch.nn.Conv2d(256, 192, 3, padding=1)
        self.lay1_h = torch.nn.Conv2d(192, 64, 3, padding=1)
        self.lay2_h = torch.nn.Conv2d(64, 8, 3, padding=1)

        self.lay_hr1_c = torch.nn.Conv2d(64, 64, 1)
        self.lay_compress = torch.nn.Conv2d(192, 64, 1)
        # self.fuse_feat = torch.nn.Conv2d(128, 64, 1)

        self.conv_norm_relus_fuse_img = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU()
        )
        
        self.conv_norm_relus_fuse_all_1 = nn.Sequential(
            nn.Conv2d(68+1, 32, 3, 1, 3, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 11, 1),
            nn.ReLU()
        )

        self.conv_norm_relus_fuse_all_2 = nn.Sequential(
            nn.Conv2d(68+1, 32, 3, 1, 5, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 11, 1),
            nn.ReLU()
        )

        self.conv_norm_relus_fuse_all_3 = nn.Sequential(
            nn.Conv2d(68+1, 32, 3, 1, 7, 7),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 11, 1),
            nn.ReLU()
        )

        self.conv_norm_relus_fuse_all_4 = nn.Sequential(
            nn.Conv2d(68+1, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 11, 1),
            nn.ReLU()
        )

        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=4)
        # used for the b4 and b4 correct; nice_light
        self.encoder = TransformerEncoder(encoder_layer, num_layers=3)


    def inference(self, samples: NestedTensor,orig_w,orig_h):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        self.object_limit = 10
        features, pos = self.detr.backbone(samples)
        
        feature_hr = features[:1]


        srcs = []
        masks = []
        poses = []
        spatial_shapes = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]
            
            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n//self.detr.num_frames, self.detr.num_frames, c, h, w)
            
            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.detr.num_frames, self.detr.num_frames, h, w)
            
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l+1].shape
            pos_l = pos[l+1].reshape(np//self.detr.num_frames, self.detr.num_frames, cp, hp, wp)
            
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        
        if self.detr.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask    # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))
                src = src.reshape(n//self.detr.num_frames, self.detr.num_frames, c, h, w)
                mask = mask.reshape(n//self.detr.num_frames, self.detr.num_frames, h, w)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np//self.detr.num_frames, self.detr.num_frames, cp, hp, wp)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = self.detr.query_embed.weight

        h_b, w_b = features[0].decompose()[0].shape[-2:]
        c_r, h_b_r, w_b_r = feature_hr[0].decompose()[0].shape[-3:]
        h_b_r = h_b_r * 2
        w_b_r = w_b_r * 2

        bottom_lvl_feature = features[0].decompose()[0].detach().clone().reshape(n//self.detr.num_frames, self.detr.num_frames, 192, h_b, w_b)
        bottom_lvl_feature_hr = F.interpolate(feature_hr[0].decompose()[0].detach(), (h_b_r, w_b_r))
        bottom_lvl_feature_hr = bottom_lvl_feature_hr.reshape(n//self.detr.num_frames, self.detr.num_frames, c_r, h_b_r, w_b_r)
        
        img_feat = F.interpolate(samples.tensors, size=(h_b_r, w_b_r))
        img_feat = self.conv_norm_relus_fuse_img(img_feat)
        img_feat = img_feat.reshape(n//self.detr.num_frames, self.detr.num_frames, 64, h_b_r, w_b_r)

        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, enc_outputs_coord_unact = self.detr.transformer(srcs, masks, poses, query_embeds)
        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        outputs_masks_inc = []
        outputs_masks_refine = []
        indices_list = []
        
        enc_lay_num = hs.shape[0]
        for lvl in range(enc_lay_num):
            if lvl < enc_lay_num -1:
                continue
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs_box[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            topkv, indices10 = torch.topk(outputs_class[0].sigmoid().cpu().detach().flatten(0),k=10) # ori is 10
            indices10 = indices10.tolist()
            sel_objs = list(set([indx // 42 for indx in indices10]))[:self.object_limit]
            
            outputs_class = outputs_class[:,sel_objs]
            outputs_coord = outputs_coord[:,:,sel_objs,:]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            
            mask_head_params = self.controller(hs[lvl][:, sel_objs])    # [bs, num_quries, num_params]
            if lvl == enc_lay_num -1:
                # print('hs[lvl] shape:', hs[lvl].shape)
                mask_head_params_inc = self.controller_inc(hs[lvl][:, sel_objs])    # [bs, num_quries, num_params]
                mask_head_params_refine = self.controller_refine(hs[lvl][:, sel_objs]) 
                

            # reference_points: [1, \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            orig_w = torch.tensor(orig_w).to(reference)
            orig_h = torch.tensor(orig_h).to(reference)
            scale_f = torch.stack([orig_w, orig_h], dim=0)
            reference_points = reference[...,:2].sigmoid()*scale_f[None,None,None,:]
            reference_points = reference_points[:,:,sel_objs]
            mask_head_params = mask_head_params
            num_insts= [len(sel_objs)] #[300]
            # mask prediction
            if lvl == enc_lay_num -1:
                outputs_layer = self.forward_mask_head_train_with_inc(outputs_layer, memory, spatial_shapes, 
                                                         reference_points, mask_head_params, mask_head_params_inc, mask_head_params_refine, num_insts, bottom_lvl_feature, bottom_lvl_feature_hr, img_feat, outputs_coord.clone()[0].permute(1, 0, 2), False)
                outputs_masks_inc.append(outputs_layer['pred_masks_inc'])
                outputs_masks_refine.append(outputs_layer['pred_masks_refine'])
            else:
                outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_points, mask_head_params, num_insts)
           
            outputs_masks.append(outputs_layer['pred_masks'])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1][0].clone()
        outputs['pred_masks_inc'] = outputs_masks_inc[-1][0].clone()
        outputs['pred_masks_refine'] = outputs_masks_refine[-1][0]
        
        return outputs


    def forward_mask_head_train_with_inc(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, mask_head_params_inc, mask_head_params_refine, num_insts, bottom_lvl_feature, bottom_lvl_feature_hr, img_feat, pred_boxes, is_train):
        bs,n_f, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        # print('spatial_shapes:', spatial_shapes)
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:,:, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.detr.num_frames, h, w, c).permute(0,4,1,2,3)
            if feat_l == 0:
                outputs['memory_feat'] = mem_l.detach()
            encod_feat_l.append(mem_l)
            spatial_indx += h * w
        
        pred_masks = []
        pred_masks_inc = []
        pred_masks_refine = []

        bottom_lvl_feature_hr = self.lay_compress(bottom_lvl_feature_hr[0]).unsqueeze(0)
        for iframe in range(self.detr.num_frames):
            bottom_lvl_feature_fi = bottom_lvl_feature[:, iframe].detach()
            bottom_lvl_feature_fi_hr = bottom_lvl_feature_hr[:, iframe].detach()
            pred_coord = pred_boxes[:, iframe].detach()
            output_boxes = pred_coord
            output_boxes[:,0::2] *= bottom_lvl_feature_fi_hr.shape[-1]
            output_boxes[:,1::2] *= bottom_lvl_feature_fi_hr.shape[-2]
            
            output_boxes[:,0] -= output_boxes[:,2] * 0.5
            output_boxes[:,1] -= output_boxes[:,3] * 0.5
            output_boxes[:,2] += output_boxes[:,0]
            output_boxes[:,3] += output_boxes[:,1]
            output_boxes = output_boxes.int()

            img_feat_fi = img_feat[:, iframe]
            # print('img feat fi shape:', img_feat_fi.shape)
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]

            
            decod_feat_f, decod_feat_f_c = self.mask_head(encod_feat_f, bottom_lvl_feature_fi, fpns=None)

            fused_x = self.lay_hr1(decod_feat_f_c)
            fused_x = F.relu(fused_x)
            fused_x = bottom_lvl_feature_fi + F.interpolate(fused_x, size=bottom_lvl_feature_fi.shape[-2:], mode="nearest")
            

            fused_x = self.lay1_h(fused_x)
            fused_x = F.relu(fused_x)
            fused_x_c = fused_x.clone()
            fused_x = self.lay2_h(fused_x)
            fused_x = F.relu(fused_x)
            # [bs, C/32, H/8, W/8]
            reference_points_i = reference_points[:,iframe]
            ######### conv ##########
            mask_logits, mask_logits_c = self.dynamic_mask_with_coords(decod_feat_f, reference_points_i, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord)
            mask_logits_inc, mask_logits_inc_c = self.dynamic_mask_with_coords_inc(fused_x, reference_points_i.detach(), mask_head_params_inc, mask_logits_c,
                                                        num_insts=num_insts,
                                                        mask_feat_stride=4,
                                                        rel_coord=self.rel_coord)

            fused_x = self.lay_hr1_c(fused_x_c)
            fused_x = F.relu(fused_x)

            mask_logits_pred_inc_hr = (F.interpolate(mask_logits_inc_c.detach().sigmoid(), size=bottom_lvl_feature_fi_hr.shape[-2:]) > 0.5).float()
            mask_logits_pred_inc_hr = mask_logits_pred_inc_hr.sum(dim=1, keepdim=True)
            mask_logits_pred_inc_hr[mask_logits_pred_inc_hr > 1.0] = 1.0

            laplacian_kernel = torch.tensor(
                [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                dtype=torch.float32, device=mask_logits_pred_inc_hr.device).reshape(1, 1, 3, 3).requires_grad_(False)

            
            boundary_targets = F.conv2d(mask_logits_pred_inc_hr, laplacian_kernel, dilation =3, padding=3)
            boundary_targets[boundary_targets > 0.5] = 1.
            boundary_targets[boundary_targets <= -0.5] = 1.
            boundary_targets = F.conv2d(boundary_targets, laplacian_kernel, dilation =3, padding=3)
            boundary_targets[boundary_targets > 0.5] = 1.
            boundary_targets[boundary_targets <= -0.5] = 1.
            point_limit = 12000
        
            fused_x_hr = F.interpolate(fused_x, size=bottom_lvl_feature_fi_hr.shape[-2:])
            fused_x_hr = fused_x_hr + boundary_targets * (bottom_lvl_feature_fi_hr + img_feat_fi)

            uncertain_pos = torch.nonzero(
                boundary_targets.squeeze(1), as_tuple=True)
            
            rand_inx = torch.randperm(len(uncertain_pos[0]))

            uncertain_pos = list(uncertain_pos)
            uncertain_pos = tuple([pos[rand_inx][:point_limit] for pos in uncertain_pos])
            

            fused_x_hr_pos = pos_embed(fused_x_hr)
            fused_x_hr_sel = fused_x_hr.clone().permute(0, 2, 3, 1)[uncertain_pos]
            fused_x_hr_sel_pos = fused_x_hr_pos.clone().permute(0, 2, 3, 1)[uncertain_pos]
        
            
            fused_x_hr_sel_encoded = self.encoder(fused_x_hr_sel.unsqueeze(1), fused_x_hr_sel_pos.unsqueeze(1))

            fused_x_hr = fused_x_hr.permute(0, 2, 3, 1)

            fused_x_hr[uncertain_pos] = fused_x_hr[uncertain_pos] + fused_x_hr_sel_encoded.squeeze(1)
            fused_x_hr = fused_x_hr.permute(0, 3, 1, 2)

            mask_logits_refine, _ = self.dynamic_mask_with_coords_inc_super(fused_x_hr, reference_points_i.detach(), mask_head_params_refine, mask_logits_inc_c, mask_logits_c, output_boxes, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=2,
                                                        rel_coord=self.rel_coord)

            mask_f = []
            mask_f_inc = []
            mask_f_refine = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                mask_f_inc.append(mask_logits_inc[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                mask_f_refine.append(mask_logits_refine[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)  
            pred_masks_inc.append(mask_f_inc)  
            pred_masks_refine.append(mask_f_refine)

        output_pred_masks = []
        output_pred_masks_inc = []
        output_pred_masks_refine = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))
            out_masks_b_inc = [m[i] for m in pred_masks_inc]
            output_pred_masks_inc.append(torch.cat(out_masks_b_inc, dim=2))
            out_masks_b_refine = [m[i] for m in pred_masks_refine]
            output_pred_masks_refine.append(torch.cat(out_masks_b_refine, dim=2))
        
        outputs['pred_masks'] = output_pred_masks
        outputs['pred_masks_inc'] = output_pred_masks_inc
        outputs['pred_masks_refine'] = output_pred_masks_refine
        return outputs

    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        bs,n_f, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:,:, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.detr.num_frames, h, w, c).permute(0,4,1,2,3)
            if feat_l == 0:
                outputs['memory_feat'] = mem_l.detach()
            encod_feat_l.append(mem_l)
            spatial_indx += h * w
        pred_masks = []
        for iframe in range(self.detr.num_frames):
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]

         
            decod_feat_f = self.mask_head(encod_feat_f, None, fpns=None)
            # [bs, C/32, H/8, W/8]
            reference_points_i = reference_points[:,iframe]
            ######### conv ##########
            mask_logits, _ = self.dynamic_mask_with_coords(decod_feat_f, reference_points_i, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)  
        
        # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))
        
        outputs['pred_masks'] = output_pred_masks
        return outputs

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts, 
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        # print('N:', N)
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        
        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
       
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )
            
            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        mask_logits_c = mask_logits.detach()
        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits_c = mask_logits_c.reshape(1, -1, mask_logits_c.shape[-2], mask_logits_c.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits, mask_logits_c
    
    def dynamic_mask_with_coords_inc(self, mask_feats, reference_points, mask_head_params, mask_logits_pred, num_insts, 
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        # print('N:', N)
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        
        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            
            mask_logits_pred = F.interpolate(mask_logits_pred.detach().sigmoid(), size=(H, W)).unsqueeze(-1)
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            mask_logits_pred = mask_logits_pred.permute(0, 1, 4, 2, 3).flatten(-2, -1)

            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_logits_pred_b = mask_logits_pred[:, inst_st: inst_st + num_inst, :, :]
                
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_logits_pred_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
        # print('mask_head_inputs shape:', mask_head_inputs.shape)
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums_inc, self.bias_nums
            )
            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        mask_logits_c = mask_logits.detach()
        # upsample predicted masks
        
        mask_logits = aligned_bilinear(mask_logits, 2)
        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits_c = mask_logits_c.reshape(1, -1, mask_logits_c.shape[-2], mask_logits_c.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits, mask_logits_c

    def dynamic_mask_with_coords_inc_super(self, mask_feats, reference_points, mask_head_params, mask_logits_pred, mask_logits_pred_sub, output_boxes, num_insts, 
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        # print('N:', N)
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        # print('num_insts:', num_insts)
        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            
            mask_logits_pred = F.interpolate(mask_logits_pred.detach().sigmoid(), size=(H, W)).unsqueeze(-1)
            mask_logits_pred_sub = F.interpolate(mask_logits_pred_sub.detach().sigmoid(), size=(H, W)).unsqueeze(-1)
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3) #.flatten(-2, -1)
            mask_logits_pred = mask_logits_pred.permute(0, 1, 4, 2, 3) #.flatten(-2, -1)
            mask_logits_pred_sub = mask_logits_pred_sub.permute(0, 1, 4, 2, 3)

            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                if num_inst > self.object_limit:
                    num_inst = self.object_limit

                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :][0]
                mask_logits_pred_b = mask_logits_pred[:, inst_st: inst_st + num_inst, :, :][0]
                mask_logits_pred_b_sub = mask_logits_pred_sub[:, inst_st: inst_st + num_inst, :, :][0]
                mask_logits_pred_b_sub_mask = torch.zeros_like(mask_logits_pred_b)
                for ii in range(output_boxes.shape[0]):
                    if ii >= self.object_limit:
                        continue
                    mask_logits_pred_b_sub_mask[ii,0, output_boxes[ii,1]:output_boxes[ii,3], output_boxes[ii,0]:output_boxes[ii,2]] = 1.0
                mask_feats_bi = mask_feats[i].unsqueeze(0).repeat(num_inst, 1, 1, 1)

                mask_feats_bi_new = torch.cat((mask_feats_bi, mask_logits_pred_b, mask_logits_pred_b_sub, mask_logits_pred_b_sub_mask, relative_coords_b), dim =1)

                mask_feats_bi_final = (self.conv_norm_relus_fuse_all_1(mask_feats_bi_new) + self.conv_norm_relus_fuse_all_2(mask_feats_bi_new) + self.conv_norm_relus_fuse_all_3(mask_feats_bi_new) + self.conv_norm_relus_fuse_all_4(mask_feats_bi_new)).unsqueeze(0)

                mask_head_inputs.append(mask_feats_bi_final)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
        # print('mask_head_params shape:', mask_head_params.shape)
        # print('mask_head_inputs shape:', mask_head_inputs.shape)
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums_inc, self.bias_nums
            )
            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        mask_logits_c = mask_logits.detach()
        # upsample predicted masks
        
        mask_logits = aligned_bilinear(mask_logits, 2)
        # print('mask logits shape af:', mask_logits.shape)
        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits_c = mask_logits_c.reshape(1, -1, mask_logits_c.shape[-2], mask_logits_c.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits, mask_logits_c


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim//4, 3, padding=1)
        self.lay2 = torch.nn.Conv2d(dim//4, dim//32, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
    
        self.dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, bottom_lvl_feature_fi, fpns):
        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]

        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)
        
        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        
        fused_x_c = fused_x.clone()

        fused_x = self.dcn(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        if bottom_lvl_feature_fi != None:
            return fused_x, fused_x_c

        return fused_x

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def dice_loss_my(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def sigmoid_focal_loss_my(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    prob = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # output single / multi frames
        assert len(orig_target_sizes) == len(max_target_sizes)
        # max_h, max_w = max_target_sizes.max(0)[0].tolist()

        # pred_logits: [bs, num_querries, num_classes]
        # pred_masks: [bs, num_querries, num_frames, H/8, W/8]

        out_refs = outputs['reference_points']
        outputs_masks = outputs["pred_masks"]
        out_logits = outputs['pred_logits']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        outputs_masks = [out_m[topk_boxes[i]].unsqueeze(0) for i, out_m in enumerate(outputs_masks)]
        outputs_masks = torch.cat(outputs_masks)
        bs, _, num_frames, H, W = outputs_masks.shape

        outputs_masks = F.interpolate(outputs_masks.flatten(0,1), size=(H*4, W*4), mode="bilinear", align_corners=False)
        outputs_masks = outputs_masks.sigmoid() > self.threshold

        # [bs, num_frames, 10, H, W]
        outputs_masks = outputs_masks.reshape(bs, -1, num_frames, outputs_masks.shape[-2], outputs_masks.shape[-1]).permute(0,2,1,3,4)

        # reference points for each instance
        references = [refs[topk_boxes[i]].unsqueeze(0) for i, refs in enumerate(out_refs)]
        references = torch.cat(references)

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["scores"] = scores[i]
            results[i]["labels"] = labels[i]
            results[i]['reference_points'] = references[i]
            
            results[i]["masks"] = cur_mask[:, :, :img_h, :img_w]
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest").byte()
            results[i]["masks"] = results[i]["masks"].permute(1,0,2,3)

        # required dim of results:
        #   scores: [num_ins]
        #   labels: [num_ins]
        #   reference_points: [num_ins, num_frames, 2]
        #   masks: [num_ins, num_frames, H, W]

        return results



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        q = k = self.with_pos_embed(src, pos)
        # q = k = src
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
