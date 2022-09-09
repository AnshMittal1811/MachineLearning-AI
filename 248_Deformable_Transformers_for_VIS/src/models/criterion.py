"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .deformable_segmentation import sigmoid_focal_loss, dice_loss
from ..util import box_ops
from ..util.misc import (nested_tensor_from_tensor_list,
                           accuracy, get_world_size, interpolate,
                           is_dist_avail_and_initialized)

AUX_LOSS_WEIGHTING_COEF = {
    5: 1 / 2,
    4: 5 / 30,
    3: 4 / 30,
    2: 3 / 30,
    1: 2 / 30,
    0: 1 / 30,
}


class SetCriterion(nn.Module):
    """ This class computes the loss for DeVIS.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_loss, focal_alpha):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha

    def _get_src_permutation_idx(self, indices, from_devis=False):
        # permute predictions following indices
        if from_devis:
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _, _) in indices])
        else:
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_masked_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)[mask] for i, (src, _, mask) in enumerate(indices)])
        src_idx = torch.cat([src[mask] for (src, _, mask) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices, from_devis=False):
        # permute targets following indices
        if from_devis:
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt, _) in enumerate(indices)])
            src_idx = torch.cat([tgt for (_, tgt, _) in indices])
        else:
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
            src_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_masked_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)[mask] for i, (_, tgt, mask) in enumerate(indices)])
        src_idx = torch.cat([tgt[mask] for (_, tgt, mask) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        raise NotImplementedError
        # """Classification loss (NLL)
        # targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        # """
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_masked_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, _, _, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o
        #
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {'loss_ce': loss_ce}
        #
        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        from_devis = len(indices[0]) == 3
        if from_devis:
            idx = self._get_src_permutation_masked_idx(indices)
            target_classes_o = torch.cat([t["labels"][J[mask]] for t, (_, J, mask) in zip(targets, indices)])

        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        from_devis = len(indices[0]) == 3
        if from_devis:
            if not torch.any(indices[0][2]):
                target_boxes = torch.cat([t['boxes'][i[mask]] for t, (_, i, mask) in zip(targets, indices)], dim=0)
                idx = self._get_src_permutation_masked_idx(indices)
            else:
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i, _) in zip(targets, indices)], dim=0)
                idx = self._get_src_permutation_idx(indices, from_devis)
        else:
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        from_devis = len(indices[0]) == 3

        if from_devis:
            if not torch.any(indices[0][2]):
                tgt_idx = self._get_tgt_permutation_masked_idx(indices)
                src_masks = src_masks[indices[0][2]]
            else:
                tgt_idx = self._get_tgt_permutation_idx(indices, from_devis)
            src_masks = src_masks.unsqueeze(1)
        else:
            tgt_idx = self._get_tgt_permutation_idx(indices)

        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets], split=False).decompose()
        target_masks = target_masks.to(src_masks)
        target_res = target_masks.shape[-2:]
        target_masks = target_masks[tgt_idx].flatten(1)

        src_masks = interpolate(src_masks, size=target_res, mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        if "indices" in outputs:
            indices = outputs["indices"]
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
            indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if "indices" in aux_outputs:
                    indices = aux_outputs["indices"]
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' and "pred_masks" not in aux_outputs:
                        # We compute aux_mask_loss if available in that level
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(matcher, num_classes, cfg):

    weight_dict = {'loss_ce': cfg.MODEL.LOSS.CLASS_COEF,
                   'loss_bbox': cfg.MODEL.LOSS.BBX_L1_COEF,
                   'loss_giou': cfg.MODEL.LOSS.BBX_GIOU_COEF, }

    # TODO this is a hack
    if cfg.MODEL.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        if cfg.MODEL.LOSS.AUX_LOSS_WEIGHTING:
            for i in range(cfg.MODEL.TRANSFORMER.DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v * AUX_LOSS_WEIGHTING_COEF[i] for k, v in weight_dict.items()})

            weight_dict['loss_ce'] *= AUX_LOSS_WEIGHTING_COEF[5]
            weight_dict['loss_bbox'] *= AUX_LOSS_WEIGHTING_COEF[5]
            weight_dict['loss_giou'] *= AUX_LOSS_WEIGHTING_COEF[5]
            weight_dict.update(aux_weight_dict)

        else:
            for i in range(cfg.MODEL.TRANSFORMER.DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    if cfg.MODEL.MASK_ON:
        losses.append('masks')
        weight_dict["loss_mask"] = cfg.MODEL.LOSS.SEGM_MASK_COEF
        weight_dict["loss_dice"] = cfg.MODEL.LOSS.SEGM_DICE_COEF
        if cfg.MODEL.LOSS.MASK_AUX_LOSS:
            for i in cfg.MODEL.LOSS.MASK_AUX_LOSS:
                weight_dict[f"loss_mask_{i}"] = cfg.MODEL.LOSS.SEGM_MASK_COEF
                weight_dict[f"loss_dice_{i}"] = cfg.MODEL.LOSS.SEGM_DICE_COEF

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_loss=cfg.MODEL.LOSS.FOCAL_LOSS,
        eos_coef=cfg.MODEL.LOSS.EOS,
        focal_alpha=cfg.MODEL.LOSS.FOCAL_ALPHA,
    )

    return criterion