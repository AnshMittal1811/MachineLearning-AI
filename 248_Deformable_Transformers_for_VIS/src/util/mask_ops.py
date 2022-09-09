from pycocotools import mask as mask_utils
import numpy as np

# From TrackEval implementation https://github.com/JonathonLuiten/TrackEval
def compute_iou_matrix(masks1, masks2, is_encoded=False, do_ioa=False):
    """ Calculates the IOU (intersection over union) between two arrays of segmentation masks.
    If is_encoded a run length encoding with pycocotools is assumed as input format, otherwise an input of numpy
    arrays of the shape (num_masks, height, width) is assumed and the encoding is performed.
    If do_ioa (intersection over area) , then calculates the intersection over the area of masks1 - this is commonly
    used to determine if detections are within crowd ignore region.
    :param masks1:  first set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                    else pycocotools rle encoded format)
    :param masks2:  second set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                    else pycocotools rle encoded format)
    :param is_encoded: whether the input is in pycocotools rle encoded format
    :param do_ioa: whether to perform IoA computation
    :return: the IoU/IoA scores
    """

    # use pycocotools for run length encoding of masks
    if not is_encoded:
        masks1 = mask_utils.encode(np.array(np.transpose(masks1, (1, 2, 0)), order='F'))
        masks2 = mask_utils.encode(np.array(np.transpose(masks2, (1, 2, 0)), order='F'))

    # use pycocotools for iou computation of rle encoded masks
    ious = np.asarray(mask_utils.iou(masks1, masks2, [do_ioa for _ in range(len(masks1))]))
    if len(masks1) == 0 or len(masks2) == 0:
        ious = ious.reshape(len(masks1), len(masks2))
    assert (ious >= 0).all()
    assert (ious <= 1).all()

    return ious