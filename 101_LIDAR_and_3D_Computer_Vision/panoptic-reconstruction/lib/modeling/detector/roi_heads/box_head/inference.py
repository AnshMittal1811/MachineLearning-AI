# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lib.structures.bounding_box import BoxList
from lib.structures.boxlist_ops import boxlist_nms, cat_boxlist
from lib.modeling.detector.box_coder import BoxCoder
from lib.config import config


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super().__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []

        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            box_list = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            box_list = box_list.clip_to_image(remove_empty=False)
            box_list = self.filter_results(box_list, num_classes)
            results.append(box_list)

        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)

        scores = scores.reshape(-1)
        box_list = BoxList(boxes, image_shape, mode="xyxy")
        box_list.add_field("scores2d", scores)

        return box_list

    def filter_results(self, box_list, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the box_list to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = box_list.bbox.reshape(-1, num_classes * 4)
        scores = box_list.get_field("scores2d").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class

        inds_all = scores > self.score_thresh

        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            box_list_for_class = BoxList(boxes_j, box_list.size, mode="xyxy")
            box_list_for_class.add_field("scores2d", scores_j)

            box_list_for_class = boxlist_nms(box_list_for_class, self.nms)
            num_labels = len(box_list_for_class)
            box_list_for_class.add_field("label", torch.full((num_labels,), j, dtype=torch.int64, device=device))
            result.append(box_list_for_class)

        result = cat_boxlist(result)
            
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores2d")
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.detections_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]

        return result


def make_roi_box_post_processor():
    bbox_reg_weights = config.MODEL.INSTANCE2D.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = config.MODEL.INSTANCE2D.ROI_HEADS.SCORE_THRESH
    nms_thresh = config.MODEL.INSTANCE2D.ROI_HEADS.NMS
    detections_per_img = config.MODEL.INSTANCE2D.ROI_HEADS.DETECTIONS_PER_IMG

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
    )
    return postprocessor
