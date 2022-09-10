# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes

    def encode_z(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1
        ex_widths = proposals[:, 1] - proposals[:, 0] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths

        gt_widths = reference_boxes[:, 1] - reference_boxes[:, 0] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dw = ww * torch.log(gt_widths / ex_widths)

        targets = torch.stack((targets_dx, targets_dw), dim=1)
        return targets

    def decode_z(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1
        widths = boxes[:, 1] - boxes[:, 0] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::2] / wx
        dw = rel_codes[:, 1::2] / ww

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_w = torch.exp(dw) * widths[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::2] = pred_ctr_x - 0.5 * pred_w
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 1::2] = pred_ctr_x + 0.5 * pred_w - 1

        return pred_boxes

    def encode_xyz(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 3] - proposals[:, 0] + TO_REMOVE
        ex_lengths = proposals[:, 1] - proposals[:, 4] + TO_REMOVE
        ex_heights = proposals[:, 5] - proposals[:, 2] + TO_REMOVE

        # clamp
        ex_widths = ex_widths.clamp(min=0.01)
        ex_lengths = ex_lengths.clamp(min=0.01)
        ex_heights = ex_heights.clamp(min=0.01)

        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 4] + 0.5 * ex_lengths
        ex_ctr_z = proposals[:, 2] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 3] - reference_boxes[:, 0] + TO_REMOVE
        gt_lengths = reference_boxes[:, 1] - reference_boxes[:, 4] + TO_REMOVE
        gt_heights = reference_boxes[:, 5] - reference_boxes[:, 2] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 4] + 0.5 * gt_lengths
        gt_ctr_z = reference_boxes[:, 2] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wx * (gt_ctr_y - ex_ctr_y) / ex_lengths
        targets_dz = wx * (gt_ctr_z - ex_ctr_z) / ex_heights

        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dl = ww * torch.log(gt_lengths / ex_lengths)
        targets_dh = ww * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dl, targets_dh), dim=1)

        return targets

    def decode_xyz(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1
        widths = boxes[:, 3] - boxes[:, 0] + TO_REMOVE
        lengths = boxes[:, 1] - boxes[:, 4] + TO_REMOVE
        heights = boxes[:, 5] - boxes[:, 2] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 4] + 0.5 * lengths
        ctr_z = boxes[:, 2] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wx
        dz = rel_codes[:, 2::6] / wx

        dw = rel_codes[:, 3::6] / ww
        dl = rel_codes[:, 4::6] / ww
        dh = rel_codes[:, 5::6] / ww

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * lengths[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * heights[:, None] + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 4::6] = pred_ctr_y - 0.5 * pred_l
        pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 1::6] = pred_ctr_y + 0.5 * pred_l - 1
        pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_h - 1

        return pred_boxes

    def encode_xyz3d(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 3] - proposals[:, 0] + TO_REMOVE
        ex_lengths = proposals[:, 4] - proposals[:, 1] + TO_REMOVE
        ex_heights = proposals[:, 5] - proposals[:, 2] + TO_REMOVE

        # clamp
        ex_widths = ex_widths.clamp(min=1)
        ex_lengths = ex_lengths.clamp(min=1)
        ex_heights = ex_heights.clamp(min=1)

        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_lengths
        ex_ctr_z = proposals[:, 2] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 3] - reference_boxes[:, 0] + TO_REMOVE
        gt_lengths = reference_boxes[:, 4] - reference_boxes[:, 1] + TO_REMOVE
        gt_heights = reference_boxes[:, 5] - reference_boxes[:, 2] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_lengths
        gt_ctr_z = reference_boxes[:, 2] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wx * (gt_ctr_y - ex_ctr_y) / ex_lengths
        targets_dz = wx * (gt_ctr_z - ex_ctr_z) / ex_heights

        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dl = ww * torch.log(gt_lengths / ex_lengths)
        targets_dh = ww * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dl, targets_dh), dim=1)

        return targets

    def decode_xyz3d(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1
        widths = boxes[:, 3] - boxes[:, 0] + TO_REMOVE
        lengths = boxes[:, 4] - boxes[:, 1] + TO_REMOVE
        heights = boxes[:, 5] - boxes[:, 2] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * lengths
        ctr_z = boxes[:, 2] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wx
        dz = rel_codes[:, 2::6] / wx

        dw = rel_codes[:, 3::6] / ww
        dl = rel_codes[:, 4::6] / ww
        dh = rel_codes[:, 5::6] / ww

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * lengths[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * heights[:, None] + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_l
        pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 4::6] = pred_ctr_y + 0.5 * pred_l - 1
        pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_h - 1

        return pred_boxes
