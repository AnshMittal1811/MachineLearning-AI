# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model
"""
import copy
import torch
import torch.nn.functional as F
from torch import nn
import math

from ..util import box_ops
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                           inverse_sigmoid)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss, with_box_refine,
                 with_ref_point_refine, with_gradient):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.with_gradient = with_gradient
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        # Allows using /32 resolution when only 1 is used
        if num_feature_levels == 1:
            num_channels = [self.backbone.num_channels[3]]
        else:
            num_channels = self.backbone.num_channels[-3:]

        if num_feature_levels > 1:
            input_proj_list = []
            num_backbone_outs = len(self.backbone.strides) - 1

            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)

        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.with_ref_point_refine = with_ref_point_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        # print(self.class_embed.weight)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

            if with_ref_point_refine:
                ref_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
                nn.init.constant_(ref_point_embed.layers[-1].weight.data, 0)
                nn.init.constant_(ref_point_embed.layers[-1].bias.data, 0)
                self.transformer.decoder.ref_point_embed = _get_clones(ref_point_embed, num_pred)

    def init_queries_for_devis(self, num_frames, use_instance_level_queries):
        if not use_instance_level_queries:
            num_trajectories = self.query_embed.shape[0] // num_frames
            new_weights = torch.empty((num_trajectories, self.transformer.d_model * 2))
            nn.init.normal_(new_weights)
            new_weights = new_weights.repeat(num_frames, 1, 1)
            with torch.no_grad():
                self.query_embed.weight = nn.Parameter(new_weights.flatten(0, 1))

    def forward(self, samples: NestedTensor, targets: dict = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        features_all = features
        if self.num_feature_levels == 1:
            features, pos = [features[-1]], [pos[-1]]

        else:
            features, pos = features[1:], pos[1:]

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj = self.input_proj[l](src)
            srcs.append(src_proj)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src_proj = self.input_proj[l](features[-1].tensors)
                else:
                    src_proj = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src_proj.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src_proj, mask)).to(src_proj.dtype)
                srcs.append(src_proj)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, query_pos, memory, init_reference, inter_references, level_start_index, valid_ratios, spatial_shapes = self.transformer(srcs, masks, pos, query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            if self.with_gradient:
                outputs_coord = inter_references[lvl]
            else:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()

            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, features_all, memory, hs, query_pos, srcs, masks, init_reference, inter_references, level_start_index, valid_ratios, spatial_shapes

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def process_boxes(boxes, target_sizes):
    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    return boxes


class DefDETRPostProcessor(nn.Module):
    def __init__(self, focal_loss, num_out, use_top_k):
        super().__init__()
        self.focal_loss = focal_loss
        self.num_out = num_out
        self.use_top_k = use_top_k

    def top_k_process_results(self, output_prob, boxes):
        scores, top_k_indexes = torch.topk(output_prob.view(output_prob.shape[0], -1), self.num_out, dim=1)
        query_top_k_indexes = torch.div(top_k_indexes, output_prob.shape[2], rounding_mode='trunc')
        labels = top_k_indexes % output_prob.shape[2]
        boxes = torch.gather(boxes, 1, query_top_k_indexes.unsqueeze(-1).repeat(1, 1, 4))
        return scores, labels, boxes, query_top_k_indexes

    def process_output(self, outputs):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        if self.focal_loss:
            output_probs = out_logits.sigmoid()
        else:
            output_probs = F.softmax(out_logits, -1)[..., :-1]

        scores, labels, boxes, query_top_k_indexes = self.top_k_process_results(output_probs, out_bbox)
        outputs["query_top_k_indexes"] = query_top_k_indexes

        outputs["pre_computed_results"] = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }
        return outputs

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = outputs['pred_logits']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if "pre_computed_results" not in outputs:
            outputs = self.process_output(outputs)

        scores = outputs["pre_computed_results"]["scores"]
        labels = outputs["pre_computed_results"]["labels"]
        boxes = outputs["pre_computed_results"]["boxes"]

        boxes = process_boxes(boxes, target_sizes)

        results = [
            {'scores': s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

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
