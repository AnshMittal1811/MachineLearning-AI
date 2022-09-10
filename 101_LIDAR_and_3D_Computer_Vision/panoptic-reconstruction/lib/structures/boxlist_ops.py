# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import math

from lib.layers.nms import nms as _box_nms
from lib.structures.bounding_box import BoxList
from lib.modeling.detector.utils import cat


# --------------------------------------------------------------
# iou functions
# --------------------------------------------------------------


def get_iou_mask2d(mask2d_a, mask2d_b):
    """ Compute IoU of two masks.
        maskA : mask (240, 320)
    """
    inter_area = np.count_nonzero(np.logical_and(mask2d_a, mask2d_b))
    union_area = np.count_nonzero(np.logical_or(mask2d_a, mask2d_b))

    if union_area == 0:
        return 0
    else:
        return inter_area / float(union_area)


def get_iou_bbox2d(bbox2d_A, bbox2d_B):
    """ Compute IoU of two bounding boxes.
    """
    # determine the (x, y, z)-coordinates of the intersection rectangle
    minx_overlap = max(bbox2d_A[0], bbox2d_B[0])
    miny_overlap = max(bbox2d_A[1], bbox2d_B[1])

    maxx_overlap = min(bbox2d_A[2], bbox2d_B[2])
    maxy_overlap = min(bbox2d_A[3], bbox2d_B[3])

    # compute the area of intersection rectangle
    interArea = max(0, maxx_overlap - minx_overlap) * max(0, maxy_overlap - miny_overlap)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bbox2d_A[2] - bbox2d_A[0]) * (bbox2d_A[3] - bbox2d_A[1])
    boxBArea = (bbox2d_B[2] - bbox2d_B[0]) * (bbox2d_B[3] - bbox2d_B[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_iou_bbox3d(bbox3d_A, bbox3d_B):
    """ Compute IoU of two bounding boxes.
    """
    # determine the (x, y, z)-coordinates of the intersection rectangle

    minx_overlap = max(bbox3d_A[0].item(), bbox3d_B[0].item())
    miny_overlap = max(bbox3d_A[1].item(), bbox3d_B[1].item())
    minz_overlap = max(bbox3d_A[2].item(), bbox3d_B[2].item())

    maxx_overlap = min(bbox3d_A[3].item(), bbox3d_B[3].item())
    maxy_overlap = min(bbox3d_A[4].item(), bbox3d_B[4].item())
    maxz_overlap = min(bbox3d_A[5].item(), bbox3d_B[5].item())

    # compute the area of intersection rectangle
    interArea = max(0, maxx_overlap - minx_overlap) * max(0, maxy_overlap - miny_overlap) * max(0,
                                                                                                maxz_overlap - minz_overlap)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bbox3d_A[3] - bbox3d_A[0]) * (bbox3d_A[4] - bbox3d_A[1]) * (bbox3d_A[5] - bbox3d_A[2])
    boxBArea = (bbox3d_B[3] - bbox3d_B[0]) * (bbox3d_B[4] - bbox3d_B[1]) * (bbox3d_B[5] - bbox3d_B[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except:
        iou = 0

    # return the intersection over union value
    return iou


def get_iou_mask3d(bbox3d_A, mask3d_A, bbox3d_B, mask3d_B):
    # blending in the same volumn with optimization
    minx_A = math.floor(bbox3d_A[0].item())
    miny_A = math.floor(bbox3d_A[1].item())
    minz_A = math.floor(bbox3d_A[2].item())

    maxx_A = math.ceil(bbox3d_A[3].item())
    maxy_A = math.ceil(bbox3d_A[4].item())
    maxz_A = math.ceil(bbox3d_A[5].item())

    minx_B = math.floor(bbox3d_B[0].item())
    miny_B = math.floor(bbox3d_B[1].item())
    minz_B = math.floor(bbox3d_B[2].item())

    maxx_B = math.ceil(bbox3d_B[3].item())
    maxy_B = math.ceil(bbox3d_B[4].item())
    maxz_B = math.ceil(bbox3d_B[5].item())

    minx = min(minx_A, minx_B)
    miny = min(miny_A, miny_B)
    minz = min(minz_A, minz_B)

    maxx = max(maxx_A, maxx_B)
    maxy = max(maxy_A, maxy_B)
    maxz = max(maxz_A, maxz_B)

    length_x = maxx - minx
    length_y = maxy - miny
    length_z = maxz - minz

    scene_A = np.zeros((length_x, length_y, length_z))
    scene_B = np.zeros((length_x, length_y, length_z))

    scene_A[minx_A - minx:maxx_A - minx, miny_A - miny:maxy_A - miny, minz_A - minz:maxz_A - minz] = mask3d_A
    scene_B[minx_B - minx:maxx_B - minx, miny_B - miny:maxy_B - miny, minz_B - minz:maxz_B - minz] = mask3d_B

    interArea = np.count_nonzero(np.logical_and(scene_A, scene_B))
    unionArea = np.count_nonzero(np.logical_or(scene_A, scene_B))

    if unionArea == 0:
        iou_3d = 0
    else:
        iou_3d = interArea / float(unionArea)

    return iou_3d


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores2d"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
            (ws >= min_size) & (hs >= min_size)
    ).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


def remove_small_boxes3d(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    z = boxlist.get_field('z')
    minx, miny, minz, maxx, maxy, maxz = z.unbind(dim=1)
    keep = (
            (maxx - minx >= min_size) &
            (miny - maxy >= min_size) &
            (maxz - minz >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist3d_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    box1 = boxlist1.get_field('bboxes3d')
    box2 = boxlist2.get_field('bboxes3d')
    N = box1.shape[0]
    M = box2.shape[0]
    iou = torch.zeros((N, M))
    for i in range(N):
        for j in range(M):
            iou[i, j] = get_iou_bbox3d(box1[i], box2[j])

    return iou


def bboxes3d_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    N = box1.shape[0]
    M = box2.shape[0]
    iou = torch.zeros((N, M))
    for i in range(N):
        for j in range(M):
            iou[i, j] = get_iou_bbox3d(box1[i], box2[j])

    return iou


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def copy_from(from_, keywords):
    detections = []
    for target in from_:
        detection_placeholder = BoxList(target.bbox, target.size, mode="xyxy")
        for keyword in keywords:
            # objectness
            if keyword == 'objectness':
                detection_placeholder.add_field('objectness', torch.ones(len(target)))

            # scores
            elif keyword == 'scores2d':
                detection_placeholder.add_field('scores2d', torch.ones(len(target)))

            # masks2d
            elif keyword == 'masks2d':
                mask2d = target.get_field('masks2d').get_mask_tensor()
                if mask2d.ndimension() == 2:
                    mask2d.unsqueeze_(0)
                detection_placeholder.add_field('masks2d', mask2d)

            # labels2d, etc
            else:
                detection_placeholder.add_field(keyword, target.get_field(keyword))

        detections.append(detection_placeholder)

    return detections
