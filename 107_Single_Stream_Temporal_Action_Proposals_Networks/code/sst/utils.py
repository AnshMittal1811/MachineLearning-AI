"""
utils.py
--------
This document contains utility functions for use with the SST model.
"""
import numpy as np

def get_segments(y, delta=16):
    """Convert predicted output tensor (y_pred) from SST model into the
    corresponding temporal proposals. Can perform standard confidence
    thresholding/post-processing (e.g. non-maximum suppression) to select
    the top proposals afterwards.

    Parameters
    ----------
    y : ndarray
        Predicted output from SST model of size (L, K), where L is the length of
        the input video in terms of discrete time steps.
    delta : int, optional
        The temporal resolution of the visual encoder in terms of frames. See
        Section 3 of the main paper for additional details.

    Returns
    -------
    props : ndarray
        Two-dimensional array of shape (num_props, 2), containing the start and
        end boundaries of the temporal proposals in units of frames.
    scores : ndarray
        One-dimensional array of shape (num_props,), containing the
        corresponding scores for each detection above.
    """
    temp_props, temp_scores = [], []
    L, K = y.shape
    for i in range(L):
        for j in range(min(i+1, K)):
            temp_props.append([delta*(i-j-1), delta*i])
            temp_scores.append(y[i, j])
    props_arr, score_arr = np.array(temp_props), np.array(temp_scores)
    # filter out proposals that extend beyond the start of the video.
    idx_valid = props_arr[:, 0] >= 0
    props, scores = props_arr[idx_valid, :], score_arr[idx_valid]
    return props, scores


def nms_detections(props, scores, overlap=0.7):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    props : ndarray
        Two-dimensional array of shape (num_props, 2), containing the start and
        end boundaries of the temporal proposals.
    scores : ndarray
        One-dimensional array of shape (num_props,), containing the corresponding
        scores for each detection above.

    Returns
    -------
    nms_props, nms_scores : ndarrays
        Arrays with the same number of dimensions as the original input, but
        with only the proposals selected after non-maximum suppression.
    """
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]
    return nms_props, nms_scores
