"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from typing import List
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import pycocotools.mask as mask_util
from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, multi_giou
from ..util.mask_ops import compute_iou_matrix

INF = 100000000


class DeVISHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_frames: int = 36, num_queries: int = 360,
                 focal_loss: bool = True, focal_alpha: float = 0.25, use_l1_distance_sum: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_out = num_queries
        self.focal_loss = focal_loss
        self.use_l1_distance_sum = use_l1_distance_sum
        self.focal_alpha = focal_alpha
        self.gamma = 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for i in range(bs):
            out_prob = outputs["pred_logits"][i]
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            num_tgt = len(tgt_ids) // self.num_frames

            if num_tgt == 0:
                # If not targets just use first instance to generate mask, we will then throw everything when computing the loss
                base_index = torch.arange(start=0, end=self.num_frames).long().to(out_prob.device)

                index_i = base_index * self.num_out
                index_j = base_index
                index_valid = torch.zeros(self.num_frames, device=out_prob.device, dtype=torch.bool)

                indices.append((index_i, index_j, index_valid))

                return indices

            tgt_valid_split = tgt_valid.reshape(num_tgt, self.num_frames)

            # Compute the classification cost.
            if self.focal_loss:
                out_prob = out_prob.sigmoid()
                neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                neg_cost_class = neg_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

                pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                pos_cost_class = pos_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

                cost_class = (pos_cost_class - neg_cost_class)

            else:
                # TODO: Compute cost with background class softmax
                out_prob = out_prob.softmax(-1)
                cost_class = -out_prob[:, tgt_ids]

            out_bbox_split = out_bbox.reshape(self.num_frames, self.num_out, out_bbox.shape[-1]).permute(1, 0, 2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt, self.num_frames, 4).unsqueeze(0)

            frame_index = torch.arange(start=0, end=self.num_frames).repeat(num_tgt).long()
            total_class_cost = cost_class[:, frame_index, tgt_ids].view(self.num_out, num_tgt, self.num_frames).mean(dim=-1)

            if self.use_l1_distance_sum:
                bbx_l1_cost = torch.cdist(out_bbox_split[:, 0].transpose(1, 0), tgt_bbox_split[0, :].transpose(1, 0), p=1)
                bbx_l1_cost = bbx_l1_cost.mean(0)
            else:
                bbx_l1_cost = (out_bbox_split - tgt_bbox_split).abs().mean((-1, -2))

            bbx_giou_cost = -1 * multi_giou(box_cxcywh_to_xyxy(out_bbox_split), box_cxcywh_to_xyxy(tgt_bbox_split)).mean(-1)

            cost = self.cost_class * total_class_cost + self.cost_bbox * bbx_l1_cost + self.cost_giou * bbx_giou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())

            index_i, index_j, index_valid = [], [], []
            for j in range(len(out_i)):
                frame_index = torch.arange(start=0, end=self.num_frames)
                index_i.append(frame_index * self.num_out + out_i[j])
                index_j.append(frame_index + tgt_i[j] * self.num_frames)

                index_valid.append(tgt_valid_split[tgt_i[j]].type(torch.bool))

            index_i = torch.cat(index_i).long()
            index_j = torch.cat(index_j).long()
            index_valid = torch.cat(index_valid)
            indices.append((index_i, index_j, index_valid))

            return indices


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if self.focal_loss:
            gamma = 2.0
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    if cfg.DATASETS.TYPE == 'vis':
        return DeVISHungarianMatcher(cost_class=cfg.MODEL.MATCHER.CLASS_COST, cost_bbox=cfg.MODEL.MATCHER.BBX_L1_COST, cost_giou=cfg.MODEL.MATCHER.BBX_GIOU_COST,
                                     num_frames=cfg.MODEL.DEVIS.NUM_FRAMES,
                                     num_queries=cfg.MODEL.NUM_QUERIES // cfg.MODEL.DEVIS.NUM_FRAMES,
                                     focal_loss=cfg.MODEL.LOSS.FOCAL_LOSS,
                                     focal_alpha=cfg.MODEL.LOSS.FOCAL_ALPHA,
                                     use_l1_distance_sum=cfg.MODEL.MATCHER.USE_SUM_L1_DISTANCE)

    else:
        return HungarianMatcher(cost_class=cfg.MODEL.MATCHER.CLASS_COST, cost_bbox=cfg.MODEL.MATCHER.BBX_L1_COST, cost_giou=cfg.MODEL.MATCHER.BBX_GIOU_COST,
                                focal_loss=cfg.MODEL.LOSS.FOCAL_LOSS, focal_alpha=cfg.MODEL.LOSS.FOCAL_ALPHA, )


class HungarianInferenceMatcher:

    def __init__(self, overlap_window: int = 2, cost_class: float = 2, cost_mask_iou: float = 6, score_cost: float = 2, center_distance_cost=0,
                 use_frame_average_iou: bool = False, use_binary_mask_iou: bool = False):
        self.overlap_w = overlap_window
        self.class_cost = cost_class
        self.mask_iou_cost = cost_mask_iou
        self.score_cost = score_cost
        self.center_distance_cost = center_distance_cost
        self.use_frame_average_iou = use_frame_average_iou
        self.use_binary_mask_iou = use_binary_mask_iou

    def compute_class_cost(self, track1: List, track2: List):
        cost_classes = []
        for t in range(self.overlap_w):
            classes_clip_1 = [track.get_last_t_result(-self.overlap_w + t, "categories") for track in track1]
            classes_clip_2 = [track.get_first_t_result(t, "categories") for track in track2]

            class_matrix = np.zeros((len(classes_clip_1), len(classes_clip_2)), dtype=np.float32)
            for idx_i, class_1 in enumerate(classes_clip_1):
                for idx_j, class_2 in enumerate(classes_clip_2):
                    # Assigns cost 1 if class equals
                    class_matrix[idx_i, idx_j] = class_1 == class_2 * 1.0

            cost_classes.append(class_matrix)
        total_cost_classes = np.stack(cost_classes, axis=0).mean(axis=0)

        return total_cost_classes

    def compute_score_cost(self, track1: List, track2: List):
        cost_score = []
        for t in range(self.overlap_w):
            scores_clip_1 = [track.get_last_t_result(-self.overlap_w + t, "scores") for track in track1]
            scores_clip_2 = [track.get_first_t_result(t, "scores") for track in track2]

            score_matrix = np.zeros((len(scores_clip_1), len(scores_clip_2)), dtype=np.float32)
            for idx_i, score_1 in enumerate(scores_clip_1):
                for idx_j, score_2 in enumerate(scores_clip_2):
                    # Assigns cost 1 if class equals
                    score_matrix[idx_i, idx_j] = abs(score_1 - score_2)

            cost_score.append(score_matrix)

        total_cost_scores = np.stack(cost_score, axis=0).mean(axis=0)
        return total_cost_scores

    def compute_center_distance_cost(self, track1: List, track2: List):
        cost_ct = []
        for t in range(self.overlap_w):
            centers_clip_1 = [track.get_last_t_result(-self.overlap_w + t, "centroid_points") for track in track1]
            centers_clip_2 = [track.get_first_t_result(t, "centroid_points") for track in track2]

            distance_matrix = np.zeros((len(centers_clip_1), len(centers_clip_2)), dtype=np.float32)
            for idx_i, center_1 in enumerate(centers_clip_1):
                for idx_j, center_2 in enumerate(centers_clip_2):
                    distance_matrix[idx_i, idx_j] = np.abs(np.array(center_1) - np.array(center_2)).mean()

            cost_ct.append(distance_matrix)

        total_cost_distances = np.stack(cost_ct, axis=0).mean(axis=0)
        return total_cost_distances

    def compute_frame_average_iou_cost(self, track1, track2):
        cost_iou = []
        for t in range(self.overlap_w):
            masks_clip_1 = [track.get_last_t_result(-self.overlap_w + t, "masks") for track in track1]
            masks_clip_2 = [track.get_first_t_result(t, "masks") for track in track2]
            if self.use_binary_mask_iou:
                iou_matrix = compute_iou_matrix(masks_clip_1, masks_clip_2, is_encoded=True)

            else:
                iou_matrix = np.zeros([len(track1), len(track2)])
                for i, j in np.ndindex(iou_matrix.shape):
                    iou_matrix[i, j] = self.soft_iou(masks_clip_1[i], masks_clip_2[j])
            cost_iou.append(iou_matrix)

        total_cost_iou = np.stack(cost_iou, axis=0).mean(axis=0)

        return total_cost_iou

    # Note that for soft_iou we need pixel probabilities, not binary masks
    @staticmethod
    def soft_iou(mask_logits1, mask_logits2):
        i, u = .0, .0
        if isinstance(mask_logits1, list):
            mask_logits1 = torch.stack(mask_logits1)
            mask_logits2 = torch.stack(mask_logits2)

        i += (mask_logits1 * mask_logits2).sum()
        u += (mask_logits1 + mask_logits2 - mask_logits1 * mask_logits2).sum().clamp(1e-6)
        iou = i / u if u > .0 else .0
        iou = iou.item()
        return iou

    @staticmethod
    def iou(track1_masks, track2_masks):
        i, u = .0, .0
        for d, g in zip(track1_masks, track2_masks):
            if d and g:
                i += mask_util.area(mask_util.merge([d, g], True))
                u += mask_util.area(mask_util.merge([d, g], False))
            elif not d and g:
                u += mask_util.area(g)
            elif d and not g:
                u += mask_util.area(d)
        if not u >= .0:
            print("UNION EQUALS 0")

        iou = i / u if u > .0 else .0
        return iou

    def compute_volumetric_iou_cost(self, track1: List, track2: List):
        ious = np.zeros([len(track1), len(track2)])
        track1_masks = [track.get_last_results(self.overlap_w, "masks") for track in track1]
        track2_masks = [track.get_first_results(self.overlap_w, "masks") for track in track2]
        track1_mask_ids = [track.get_mask_id() for track in track1]
        track2_mask_ids = [track.get_mask_id() for track in track2]
        iou_func = self.iou if self.use_binary_mask_iou else self.soft_iou

        iou_values_dict = {}
        for i, j in np.ndindex(ious.shape):
            combination_hash = f"{track1_mask_ids[i]}_{track2_mask_ids[j]}"
            if combination_hash not in iou_values_dict:
                iou_value = iou_func(track1_masks[i], track2_masks[j])
                ious[i, j] = iou_value
                iou_values_dict[combination_hash] = iou_value
            else:
                ious[i, j] = iou_values_dict[combination_hash]

        return ious

    def __call__(self, track1: List, track2: List):

        if self.use_frame_average_iou:
            total_cost_iou = self.compute_frame_average_iou_cost(track1, track2)
        else:
            total_cost_iou = self.compute_volumetric_iou_cost(track1, track2)

        cost = -1 * total_cost_iou * self.mask_iou_cost

        if self.class_cost:
            total_cost_classes = self.compute_class_cost(track1, track2)
            cost += -1 * total_cost_classes * self.class_cost

        if self.score_cost:
            total_cost_scores = self.compute_score_cost(track1, track2)
            cost += total_cost_scores * self.score_cost

        if self.center_distance_cost:
            total_cost_ct = self.compute_center_distance_cost(track1, track2)
            cost += self.center_distance_cost * total_cost_ct

        track1_ids, track2_ids = linear_sum_assignment(cost)

        return track1_ids, track2_ids


def build_inference_matcher(cfg):
    return HungarianInferenceMatcher(cost_mask_iou=cfg.TEST.CLIP_TRACKING.MASK_COST,
                                     cost_class=cfg.TEST.CLIP_TRACKING.CLASS_COST,
                                     score_cost=cfg.TEST.CLIP_TRACKING.SCORE_COST,
                                     center_distance_cost=cfg.TEST.CLIP_TRACKING.CENTER_COST,
                                     overlap_window=cfg.MODEL.DEVIS.NUM_FRAMES - cfg.TEST.CLIP_TRACKING.STRIDE,
                                     use_binary_mask_iou=cfg.TEST.CLIP_TRACKING.USE_BINARY_MASK_IOU,
                                     use_frame_average_iou=cfg.TEST.CLIP_TRACKING.USE_FRAME_AVERAGE_IOU,
                                     )
