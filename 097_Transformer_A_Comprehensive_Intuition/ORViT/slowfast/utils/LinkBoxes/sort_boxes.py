import numpy as np
from numpy.lib import save
from .sort import Sort
import torch
from slowfast.utils import box_ops
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

def sort_boxes(vid_boxes, O, with_score=True):
    """
    Args:
        vid_boxes (List[List[List[int]]]): [[fid_1 boxes] p[]]
        O (int): max objects
    Return:
        np.ndarray: O * T * 4
    """
    assert with_score
    global2local = {}
    def getidx(gidx):
        if gidx not in global2local:
            global2local[gidx] = len(global2local)
        return global2local[gidx]
    vgetidx = np.vectorize(getidx)

    osort = Sort(clear_dups_threshold=-1)
    out_boxes = np.zeros([len(vid_boxes), O, 4])

    for fidx, boxes in enumerate(vid_boxes):
        if len(boxes) == 0:
            boxes = np.empty([0,5])
        boxes = np.array(boxes)
        boxes = box_ops.remove_empty_boxes(boxes)
        # boxes = osort.update(boxes)
        boxes = osort.update(boxes)
        if len(boxes) == 0:
            continue
        cboxes, iboxes = boxes[:,:4], vgetidx(boxes[:,-1].astype(np.uint64))
        mask = iboxes < out_boxes.shape[1]
        cboxes, iboxes = cboxes[mask], iboxes[mask]
        out_boxes[fidx, iboxes, :] = cboxes
    return out_boxes.transpose([1,0,2]) # O, T, 4




def sort_boxes_sorted(vid_boxes, O, saved_indices=[]):
    """
    Args:
        vid_boxes (List[np.array([n,5])]): [[xyxyxid]]
        O (int): max objects
    Return:
        np.ndarray: O * T * 4
    """
    global2local = {idx:i for i , idx in enumerate(sorted(saved_indices))}
    def getidx(gidx):
        if gidx not in global2local:
            global2local[gidx] = len(global2local)
        return global2local[gidx]
    vgetidx = np.vectorize(getidx)

    out_boxes = np.zeros([len(vid_boxes), O, 4])

    for fidx, boxes in enumerate(vid_boxes):
        boxes = np.array(boxes)
        boxes = box_ops.remove_empty_boxes(boxes)
        if len(boxes) == 0:
            continue
        cboxes, iboxes = boxes[:,:4], vgetidx(boxes[:,-1].astype(np.uint64))
        mask = iboxes < out_boxes.shape[1]
        cboxes, iboxes = cboxes[mask], iboxes[mask]
        out_boxes[fidx, iboxes, :] = cboxes
    return out_boxes.transpose([1,0,2]) # O, T, 4


