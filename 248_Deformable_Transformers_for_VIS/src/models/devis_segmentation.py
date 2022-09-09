from __future__ import annotations
from typing import List
import torch
from torch import nn
from .deformable_segmentation import DefDETRSegmBase, MaskHeadConv
from .deformable_detr import DeformableDETR
from .matcher import DeVISHungarianMatcher
from ..util.misc import NestedTensor
from ..util import box_ops



class DeVIS(DefDETRSegmBase):

    def __init__(self, defdetr: DeformableDETR, matcher: DeVISHungarianMatcher, mask_head_used_features: List[List[str]],
                 att_maps_used_res: List[str], use_deformable_conv: bool,
                 post_processor: DeVISPostProcessor, mask_aux_loss: List, num_frames: int, add_3d_conv_head: bool):

        super().__init__(defdetr, matcher, mask_head_used_features, att_maps_used_res,
                         use_deformable_conv, post_processor, mask_aux_loss)
        self.num_frames = num_frames

        # Only used for ablation studies
        self.conv_head_3d = None
        if add_3d_conv_head:
            feats_dims = self._get_mask_head_dims()
            hidden_dim, nheads = self.def_detr.transformer.d_model, self.def_detr.transformer.nhead
            num_levels = len(feats_dims)
            self.mask_head = MaskHeadConv(hidden_dim, feats_dims, nheads, use_deformable_conv,
                                          self.att_maps_used_res, num_levels, out_layer=False)

            self.conv_head_3d = build_3D_conv_head(hidden_dim // 16)

    def _expand_func_mask_head(self, tensor, lengths):
        return tensor.repeat(lengths, 1, 1, 1)

    def _get_training_embeddings(self, out, targets, hs, lvl):
        outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
        indices = self.matcher(outputs_without_aux, targets)
        out["indices"] = indices

        matched_embds_idx, _, _ = indices[0]
        matched_embds = hs[lvl][0, matched_embds_idx].view(matched_embds_idx.shape[0] // self.num_frames, self.num_frames, -1).transpose(0, 1)

        return matched_embds

    def _get_eval_embeddings(self, out, targets, hs):
        process_boxes = targets["process_boxes"] if "process_boxes" in targets else True
        top_k_idxs, results = self.postprocessor(out, targets["tgt_size"], targets["clip_length"], process_boxes)
        results["top_k_idxs"] = top_k_idxs
        top_k_embeddings, inverse_idxs = torch.unique(top_k_idxs, return_inverse=True)
        num_trajectories = top_k_embeddings.shape[0]
        hs_f = hs[:, top_k_embeddings]

        return hs_f, num_trajectories, results, inverse_idxs

    def _module_inference(self, hs_f, memories_att_map_f, masks_att_map, mask_head_feats_f, num_trajectories):
        bbox_mask_f = self.bbox_attention(hs_f, memories_att_map_f, mask=masks_att_map)
        bbox_mask_flattened = [bbox_mask.transpose(1, 0).flatten(0, 1) for bbox_mask in bbox_mask_f]
        seg_masks_f = self.mask_head(mask_head_feats_f, bbox_mask_flattened, instances_per_batch=num_trajectories, expand_func=self._expand_func_mask_head)
        if self.conv_head_3d is not None:
            seg_masks_f = seg_masks_f.view(
                (num_trajectories, self.num_frames) + seg_masks_f.shape[1:]).transpose(2, 1)
            seg_masks_f = self.conv_head_3d(seg_masks_f)[:, 0]

        outputs_seg_masks = seg_masks_f.view((num_trajectories, self.num_frames) + seg_masks_f.shape[2:])
        return outputs_seg_masks

    def _training_forward(self, targets, out, hs, loss_lvl, memories_att_map, masks_att_map, mask_head_feats):
        matched_embds = self._get_training_embeddings(out, targets, hs, loss_lvl)
        num_trajectories = matched_embds.shape[1]
        outputs_seg_masks = self._module_inference(matched_embds, memories_att_map, masks_att_map, mask_head_feats, num_trajectories)
        out["pred_masks"] = outputs_seg_masks.flatten(0, 1)

    def _inference_forward(self, targets, out, hs, memories_att_map, masks_att_map, mask_head_feats, spatial_shapes):
        num_trajectories = self.def_detr.num_queries // self.num_frames
        hs = hs[-1][0].view(self.num_frames, num_trajectories, hs.shape[-1])
        eval_embeddings, num_trajectories, results, inverse_idxs = self._get_eval_embeddings(out, targets, hs)
        outputs_seg_masks = self._module_inference(eval_embeddings, memories_att_map, masks_att_map, mask_head_feats, num_trajectories)
        out_masks = outputs_seg_masks.transpose(0, 1)
        results["masks"] = out_masks[:targets["clip_length"]]
        results["inverse_idxs"] = inverse_idxs
        results["spatial_shapes"] = spatial_shapes

        return results

    def forward(self, samples: NestedTensor, targets: list = None):
        out, hs, memories_att_map, mask_head_feats, masks_att_map, spatial_shapes = super().forward(samples, targets)
        # image level processing using box attention
        memories_att_map = [memory[0].transpose(0, 1) for memory in memories_att_map]

        mask_head_feats_new = []
        for feat in mask_head_feats:
            if feat.shape[0] == 1:
                mask_head_feats_new.append(feat[0].transpose(0, 1))
            else:
                mask_head_feats_new.append(feat)

        if self.training:
            loss_levels = [-1] + self.mask_aux_loss
            for loss_lvl in loss_levels:
                out_lvl = out if loss_lvl == -1 else out["aux_outputs"][loss_lvl]
                self._training_forward(targets, out_lvl, hs, loss_lvl, memories_att_map, masks_att_map, mask_head_feats_new)
            return out, None

        else:
            return self._inference_forward(targets, out, hs, memories_att_map, masks_att_map, mask_head_feats_new, spatial_shapes)


class DeVISPostProcessor(nn.Module):

    def __init__(self, focal_loss, num_out, use_top_k, num_frames):
        super().__init__()
        self.focal_loss = focal_loss
        self.num_out = num_out
        self.use_top_k = use_top_k
        self.num_frames = num_frames

    def process_boxes(self, boxes, tgt_size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes.cpu() * scale_fct[:, None, :]
        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], 0, img_w.item())
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], 0, img_h.item())
        boxes = boxes.reshape(self.num_frames, -1, boxes.shape[-1]).detach()
        return boxes

    @torch.no_grad()
    def forward(self, outputs, tgt_size, video_length, process_boxes=True):
        # end of model inference
        if self.focal_loss:
            logits = outputs['pred_logits'].sigmoid()
        else:
            logits = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        num_trajectories = outputs['pred_boxes'].shape[1] // self.num_frames
        assert self.focal_loss
        pred_logits = logits.reshape(self.num_frames, num_trajectories, logits.shape[-1])
        traj_probs = pred_logits[:video_length].transpose(0, 1).mean(1).flatten()

        _, top_k_indexes = torch.topk(traj_probs, self.num_out, dim=0)
        query_top_k_indexes = torch.div(top_k_indexes, logits.shape[2], rounding_mode='trunc')

        labels = top_k_indexes % logits.shape[2]
        pred_classes = labels.repeat(video_length, 1) + 1
        pred_scores = pred_logits[:, query_top_k_indexes, labels]

        ct_points = outputs["pred_boxes"][0, :, :2].clone()
        ct_points = ct_points.view(self.num_frames, num_trajectories, ct_points.shape[-1])
        pred_ct_points = ct_points[:, query_top_k_indexes]
        results = {
            "scores": pred_scores[:video_length],
            "labels": pred_classes[:video_length],
            "center_points": pred_ct_points[:video_length]
        }

        if process_boxes:
            boxes = outputs['pred_boxes'][0]
            pred_boxes = self.process_boxes(boxes, tgt_size)
            pred_boxes = pred_boxes[:, query_top_k_indexes]
            results["boxes"] = pred_boxes[:video_length]

        return query_top_k_indexes, results


def build_3D_conv_head(mask_head_hidden_size):
    return nn.Sequential(
        nn.Conv3d(mask_head_hidden_size, 12, 3, padding=2, dilation=2),
        nn.GroupNorm(4, 12),
        nn.ReLU(),
        nn.Conv3d(12, 12, 3, padding=2, dilation=2),
        nn.GroupNorm(4, 12),
        nn.ReLU(),
        nn.Conv3d(12, 12, 3, padding=2, dilation=2),
        nn.GroupNorm(4, 12),
        nn.ReLU(),
        nn.Conv3d(12, 1, 1))


# class MaskHeadConvAblation(MaskHeadConv):
#     def __init__(self, dim, fpn_dims, nheads, use_deformable_conv, multi_scale_att_maps):
#         super().__init__(dim, fpn_dims, nheads, use_deformable_conv, multi_scale_att_maps,
#                          out_layer=False)


