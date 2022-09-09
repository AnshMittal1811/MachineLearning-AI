from __future__ import annotations
import random
from abc import ABC, abstractmethod
import warnings
from typing import List, Union
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F

from ..util.misc import NestedTensor
from .matcher import HungarianMatcher
from .deformable_detr import DefDETRPostProcessor, DeformableDETR
# To circumvent cyclic import from typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .devis_segmentation import DeVISPostProcessor

res_to_idx = {
    "/64": 3,
    "/32": 2,
    "/16": 1,
    "/8": 0,
}

backbone_res_to_idx = {
    "/32": 3,
    "/16": 2,
    "/8": 1,
    "/4": 0,
}


class DefDETRSegmBase(nn.Module, ABC):

    def __init__(self, defdetr: DeformableDETR, matcher: HungarianMatcher, mask_head_used_features: List[List[str]], att_maps_used_res: List[str], use_deformable_conv: bool,
                 post_processor: Union[DefDETRPostProcessor, DeVISPostProcessor], mask_aux_loss: List):

        super().__init__()
        self.def_detr = defdetr
        self.mask_aux_loss = mask_aux_loss
        self.matcher = matcher
        self.mask_head_used_features = mask_head_used_features
        self.att_maps_used_res = att_maps_used_res
        self.postprocessor = post_processor

        self._sanity_check()

        feats_dims = self._get_mask_head_dims()

        hidden_dim, nheads = self.def_detr.transformer.d_model, self.def_detr.transformer.nhead

        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=len(self.att_maps_used_res))

        num_levels = len(feats_dims) + 1
        self.mask_head = MaskHeadConv(hidden_dim, feats_dims, nheads, use_deformable_conv, self.att_maps_used_res, num_levels)

    def _sanity_check(self):
        init_mask_head_res, init_att_map_res = self.mask_head_used_features[0][0], self.att_maps_used_res[0]
        assert init_mask_head_res == init_att_map_res, f"Starting resolution for the mask_head_used features and att_maps_used_res has to be " \
                                                       f"the same. Got {init_mask_head_res} and {init_att_map_res} respectively"

    def _get_mask_head_dims(self):
        ch_dict_en = {
            "/64": 256,
            "/32": self.def_detr.backbone.num_channels[3],
            "/16": self.def_detr.backbone.num_channels[2],
            "/8": self.def_detr.backbone.num_channels[1],
            "/4": self.def_detr.backbone.num_channels[0],
        }

        feats_dims = []
        for res, name in self.mask_head_used_features[1:]:
            if name == "backbone":
                feats_dims.append(ch_dict_en[res])
            else:
                feats_dims.append(self.def_detr.transformer.d_model)
        return feats_dims

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
        src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx

    @abstractmethod
    def _expand_func_mask_head(self, tensor, lengths):
        pass

    def _get_features_for_mask_head(self, backbone_feats: List[Tensor], srcs: List[Tensor], memories: List[Tensor]):
        features_used = []
        for res, feature_type in self.mask_head_used_features:
            if feature_type == "backbone":
                if res == "/64":
                    warnings.warn("/64 feature map is only generated for encoded and compressed backbone feats. Using the compressed one")
                    features_used.append(srcs[res_to_idx[res]])
                else:
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)

            elif feature_type == "compressed_backbone":
                if res == "/4":
                    warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                else:
                    features_used.append(srcs[res_to_idx[res]])

            elif feature_type == "encoded":
                if len(memories) == 1:
                    features_used.append(memories[0])
                else:
                    if res == "/4":
                        warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                        features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                    else:
                        features_used.append(memories[res_to_idx[res]])
            else:
                raise ValueError(
                    f"Selected feature type {feature_type} is not available. Available ones: [backbone, compressed_backbone, encoded]")
        return features_used

    def forward(self, samples: NestedTensor, targets: list):
        out, backbone_feats, memories, hs, query_pos, srcs, masks, init_reference, inter_references, \
        level_start_index, valid_ratios, spatial_shapes = self.def_detr(samples, targets)

        if len(memories) != 1:
            memories_att_map = [memories[res_to_idx[res]] for res in self.att_maps_used_res]
            masks_att_map = [masks[res_to_idx[res]] for res in self.att_maps_used_res]
        else:
            memories_att_map = [memories[0]]
            masks_att_map = [masks[0]]
        mask_head_feats = self._get_features_for_mask_head(backbone_feats, srcs, memories)

        return out, hs, memories_att_map, mask_head_feats, masks_att_map, spatial_shapes


class DeformableDETRSegm(DefDETRSegmBase):

    def _expand_func_mask_head(self, tensor, lengths):
        if isinstance(lengths, list):
            tensors = []
            for idx, length_to_repeat in enumerate(lengths):
                tensors.append(tensor[idx].unsqueeze(0).repeat(1, int(length_to_repeat), 1, 1, 1).flatten(0, 1))
            return torch.cat(tensors, dim=0)
        else:
            return tensor.unsqueeze(1).repeat(1, int(lengths), 1, 1, 1).flatten(0, 1)


    @staticmethod
    def get_src_permutation_idx(indices: List[Tensor]):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
        src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx

    @staticmethod
    def tmp_batch_fill(num_embd: int, matched_indices: List[Tensor]):
        new_indices = []
        max_num = max([idx[0].shape[0] for idx in matched_indices])
        all_pos = set(range(0, num_embd))
        for idx, (embd_idxs, _) in enumerate(matched_indices):
            num_to_fill = max_num - len(embd_idxs)
            if num_to_fill > 0:
                batch_ids = set(embd_idxs.tolist())
                unmatched_embds = random.choices(list(all_pos.difference(batch_ids)), k=num_to_fill)
                new_embd_idxs = torch.cat([embd_idxs, torch.tensor(unmatched_embds, dtype=torch.int64)])
                new_indices.append(new_embd_idxs)
            else:
                new_indices.append(embd_idxs)

        return new_indices

    def _get_matched_with_filled_embeddings(self, indices: List[Tensor], hs: Tensor, lvl: int):
        instances_per_batch = [idx[0].shape[0] for idx in indices]
        filled_indices = self.tmp_batch_fill(hs.shape[2], indices)
        num_filled_instances = len(filled_indices[0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[lvl][matched_indices].view(hs.shape[1], num_filled_instances, hs.shape[-1])
        return matched_embeddings, instances_per_batch

    def _get_training_embeddings(self, out, targets, hs, loss_lvl):
        outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
        indices = self.matcher(outputs_without_aux, targets)
        out["indices"] = indices
        matched_embeddings, instances_per_batch = self._get_matched_with_filled_embeddings(indices, hs, loss_lvl)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)

        return matched_embeddings, indices_to_pick, instances_per_batch

    def _get_eval_embeddings(self, out, hs):

        out_processed = self.postprocessor.process_output(out)
        query_top_k_indexes = out_processed["query_top_k_indexes"]
        objs_embeddings = torch.gather(hs[-1], 1, query_top_k_indexes.unsqueeze(-1).repeat(1, 1, hs.shape[-1]))

        instances_per_batch = objs_embeddings.shape[1]

        return objs_embeddings, instances_per_batch, out_processed

    def _module_inference(self, matched_embeddings, memories_att_map, masks_att_map, mask_head_feats, indices_to_pick, instances_per_batch):
        # Module inference
        bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
        # We need to remove padded instances during training
        if indices_to_pick is not None:
            bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]
        else:
            bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        return self.mask_head(mask_head_feats, bbox_masks, instances_per_batch=instances_per_batch, expand_func=self._expand_func_mask_head)

    def _training_forward(self, targets, out, hs, loss_lvl, memories_att_map, masks_att_map, mask_head_feats):

        matched_embeddings, indices_to_pick, instances_per_batch = self._get_training_embeddings(out, targets, hs, loss_lvl)
        out["pred_masks"] = self._module_inference(matched_embeddings, memories_att_map, masks_att_map, mask_head_feats, indices_to_pick, instances_per_batch)

        return out

    def _inference_forward(self, out, hs, memories_att_map, masks_att_map, mask_head_feats):
        eval_embeddings, instances_per_batch, out_processed = self._get_eval_embeddings(out, hs)
        masks = self._module_inference(eval_embeddings, memories_att_map, masks_att_map, mask_head_feats, None, instances_per_batch)
        out["pre_computed_results"]["masks"] = masks.view((hs.shape[1], instances_per_batch) + masks.shape[-2:])

        return out

    def forward(self, samples: NestedTensor, targets: list = None):
        out, hs, memories_att_map, mask_head_feats, masks_att_map, spatial_shapes = super().forward(samples, targets)

        if self.training:
            loss_levels = [-1] + self.mask_aux_loss
            for loss_lvl in loss_levels:
                out_lvl = out if loss_lvl == -1 else out["aux_outputs"][loss_lvl]
                self._training_forward(targets, out_lvl, hs, loss_lvl, memories_att_map, masks_att_map, mask_head_feats)
            return out, None

        else:
            # We need to first compute masks resulting from the hungarian matching, as we compute validation loss
            self._training_forward(targets, out, hs, -1, memories_att_map, masks_att_map, mask_head_feats)
            # We then compute inference masks, which are the ones resulting from top_k over the output scores
            return self._inference_forward(out, hs, memories_att_map, masks_att_map, mask_head_feats), None


class ModulatedDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ModulatedDeformableConv2d, self).__init__()

        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                        padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                                          padding=self.padding, mask=modulator)
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)

class MultiScaleMHAttentionMap(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, num_levels, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        for i in range(num_levels):
            layer_name = "" if i == 0 else f"_{i}"
            setattr(self, f"q_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            setattr(self, f"k_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            nn.init.zeros_(getattr(self, f"k_linear{layer_name}").bias)
            nn.init.zeros_(getattr(self, f"q_linear{layer_name}").bias)
            nn.init.xavier_uniform_(getattr(self, f"k_linear{layer_name}").weight)
            nn.init.xavier_uniform_(getattr(self, f"q_linear{layer_name}").weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def _check_input(self, k, mask):
        assert len(k) == self.num_levels
        if mask is not None:
            assert len(mask) == self.num_levels

    def forward(self, q, k, mask=None):
        self._check_input(k, mask)
        out_multi_scale_maps = []

        for i, k_lvl in enumerate(k):
            layer_name = "" if i == 0 else f"_{i}"
            q_lvl = q
            q_lvl = getattr(self, f"q_linear{layer_name}")(q_lvl)
            k_lvl = F.conv2d(k_lvl, getattr(self, f"k_linear{layer_name}").weight.unsqueeze(-1).unsqueeze(-1),
                             getattr(self, f"k_linear{layer_name}").bias)
            qh_lvl = q_lvl.view(q_lvl.shape[0], q_lvl.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
            kh_lvl = k_lvl.view(k_lvl.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k_lvl.shape[-2], k_lvl.shape[-1])
            weights = torch.einsum("bqnc,bnchw->bqnhw", qh_lvl * self.normalize_fact, kh_lvl)
            if mask is not None:
                weights.masked_fill_(mask[i].unsqueeze(1).unsqueeze(1), float("-inf"))
            weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
            # weights = self.dropout(weights)
            out_multi_scale_maps.append(weights)

        return out_multi_scale_maps


class MaskHeadConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, nheads, use_deformable_conv, multi_scale_att_maps,
                 num_levels, out_layer=True):
        super().__init__()

        out_dims = [dim // (2 ** exp) for exp in range(num_levels + 2)]
        in_dims = [dim // (2 ** exp) for exp in range(num_levels + 2)]
        for i in range(len(multi_scale_att_maps)):
            in_dims[i] += nheads

        self.multi_scale_att_maps = len(multi_scale_att_maps) > 1
        conv_layer = ModulatedDeformableConv2d if use_deformable_conv else Conv2d

        self.lay1 = conv_layer(in_dims[0], in_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, in_dims[0])

        self.lay2 = conv_layer(in_dims[0], out_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, out_dims[1])

        for i in range(1, len(fpn_dims) + 1):
            setattr(self, f"lay{i + 2}", conv_layer(in_dims[i], out_dims[i + 1], 3, padding=1))
            setattr(self, f"gn{i + 2}", torch.nn.GroupNorm(8, out_dims[i + 1]))
            setattr(self, f"adapter{i}", Conv2d(fpn_dims[i - 1], out_dims[i], 1, padding=0))

        self.out_lay = None
        if out_layer:
            self.out_lay = conv_layer(out_dims[i + 1], 1, 3, padding=1)

    def forward(self, features, bbox_mask, instances_per_batch, expand_func):

        expanded_feats = expand_func(features[0], instances_per_batch)
        x = torch.cat([expanded_feats, bbox_mask[0]], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        for lvl, feature in enumerate(features[1:]):
            cur_fpn = getattr(self, f"adapter{lvl + 1}")(feature)
            cur_fpn = expand_func(cur_fpn, instances_per_batch)
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            if self.multi_scale_att_maps and lvl + 1 < len(bbox_mask):
                x = torch.cat([x, bbox_mask[lvl + 1]], 1)
            x = getattr(self, f"lay{lvl + 3}")(x)
            x = getattr(self, f"gn{lvl + 3}")(x)
            x = F.relu(x)

        if self.out_lay is not None:
            x = self.out_lay(x)

        return x


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


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, reduce=True):
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
    if reduce:
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.mean(1).sum()


class DefDETRSegmPostProcess(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pre_computed_results"]["masks"]

        outputs_masks = F.interpolate(
            outputs_masks,
            size=(max_h, max_w),
            mode="bilinear",
            align_corners=False)

        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        zip_iter = zip(outputs_masks, max_target_sizes, orig_target_sizes)
        for i, (cur_mask, t, tt) in enumerate(zip_iter):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            )

        return results
