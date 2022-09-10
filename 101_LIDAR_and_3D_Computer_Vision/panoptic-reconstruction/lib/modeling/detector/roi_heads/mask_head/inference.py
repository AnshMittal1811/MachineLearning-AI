# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from datetime import datetime

import torch
from torch import nn

from lib.config import config
from lib.layers.misc import interpolate
from lib.structures.bounding_box import BoxList


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super().__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks corresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("label") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob, masks_raw = self.masker(mask_prob, boxes)

        results = []
        for prob, raw, box in zip(mask_prob, masks_raw, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask2d", prob[:,0,:,:])
            bbox.add_field("mask2draw", raw[:,0,:,:])
            results.append(bbox)

        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))

    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    mask = mask.float()
    box = box.float()

    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask_thresholded = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask_thresholded = (mask * 255).to(torch.uint8)

    im_mask_raw = copy_mask_pixels(box, im_h, im_w, mask)
    im_mask = copy_mask_pixels(box, im_h, im_w, mask_thresholded)

    return im_mask, im_mask_raw


def copy_mask_pixels(box, im_h, im_w, mask):
    im_mask = torch.zeros((im_h, im_w), dtype=torch.float)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    return im_mask


class Masker:
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]

        if len(res) > 0:
            res = list(zip(*res))

            res_thresholded = res[0]
            res_raw = res[1]

            if len(res_thresholded) > 0:
                res_thresholded = torch.stack(res_thresholded, dim=0)[:, None]
                res_raw = torch.stack(res_raw, dim=0)[:, None]

            return res_thresholded, res_raw

        res_thresholded = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        res_raw = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))

        return res_thresholded, res_raw

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        results = []
        results_raw = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result, result_raw = self.forward_single_image(mask, box)
            results.append(result)
            results_raw.append(result_raw)
        return results, results_raw


def make_roi_mask_post_processor():
    mask_threshold = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
    masker = Masker(threshold=mask_threshold, padding=1)
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
