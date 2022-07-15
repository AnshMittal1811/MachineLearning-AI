import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def plane_seperation(plane, min_count=500):
    """
    Convert a plane map in CHW format into:
        planes_mask: NHW, a binary map containing N valid plane (in descending order)
        planes_params: NC, the parameter of the N valid plane
    """
    C, H, W = plane.shape

    # Hashing
    if C == 1:
        plane_ = (plane[0]*1000).int()  # Horizontal plane
    elif C == 3:
        plane_ = (torch.atan2(plane[0], plane[1])*1e6).int() + (plane[2]*1000).int()  # Vertical plane
        plane_[plane.abs().sum(0) == 0] = 0
    else:
        raise NotImplementedError()

    # Detect unique planes
    val, idx, cnt = torch.unique(plane_, sorted=False, return_inverse=True, return_counts=True)

    # Aggregate results
    plane_HWC = plane.permute(1, 2, 0)
    planes_mask = []
    planes_params = []
    planes_cnt = []
    for v, c in zip(val, cnt):
        if v == 0 or c < min_count:
            continue
        cur_mask = (plane_==v)
        planes_mask.append(cur_mask)
        planes_params.append(plane_HWC[cur_mask].mean(0))
        planes_cnt.append(c)

    planes_mask = torch.stack(planes_mask, 0)
    planes_params = torch.stack(planes_params, 0)
    planes_sort_idx = torch.argsort(torch.stack(planes_cnt), descending=True)

    planes_mask = planes_mask[planes_sort_idx]
    planes_params = planes_params[planes_sort_idx]
    return planes_mask, planes_params


class MatchSegmentation(nn.Module):
    def __init__(self):
        super(MatchSegmentation, self).__init__()

    def forward(self, segmentation, prob, gt_instance, gt_plane_num):
        """
        greedy matching
        match segmentation with ground truth instance
        :param segmentation: tensor with size (N, K)
        :param prob: tensor with size (N, 1)
        :param gt_instance: tensor with size (21, h, w)
        :param gt_plane_num: int
        :return: a (K, 1) long tensor indicate closest ground truth instance id, start from 0
        """

        n, k = segmentation.size()
        _, h, w = gt_instance.size()
        assert (prob.size(0) == n and h*w  == n)

        # ingnore non planar region
        gt_instance = gt_instance[:gt_plane_num, :, :].view(1, -1, h*w)     # (1, gt_plane_num, h*w)
        segmentation = segmentation.t().view(k, 1, h*w)                     # (k, 1, h*w)

        # calculate instance wise cross entropy matrix (K, gt_plane_num)
        gt_instance = gt_instance.type(torch.float32)
        ce_loss = - (gt_instance * torch.log(segmentation + 1e-6) +
            (1-gt_instance) * torch.log(1-segmentation + 1e-6))             # (k, gt_plane_num, k*w)
        ce_loss = torch.mean(ce_loss, dim=2)                                # (k, gt_plane_num)
        matching = torch.argmin(ce_loss, dim=1, keepdim=True)

        return matching


def u_grid(B, C, H_, W):
    us = torch.arange(W).float()
    us = (us + 0.5) / W * 2 * np.pi
    us = us.reshape(1,1,1,W).repeat(B, C, H_, 1)    # B, 1, H_, W
    us = torch.atan2(torch.cos(us), torch.sin(us))  # Transpose axis
    return us

def v_grid(B, C, H_, W):
    H = W // 2
    assert (H - H_) % 2 == 0
    crop = (H - H_) // 2
    vs = torch.arange(H).float()
    vs = -((vs + 0.5) / H - 0.5) * np.pi
    if crop > 0:
        vs = vs[crop:-crop]
    vs = vs.reshape(1,1,H_,1).repeat(B, C, 1, W)
    return vs

def vplane_2_vparam(vplane):
    """
    Input:
        vplane: [B, 3, H, W]. The parameters <a, b, c> for line ax + by + c = 0

    Return:
        vparam: [B, 2, H, W]. The shortes vector from camera point to the plane
    """
    vparam = torch.cross(
        vplane,
        torch.stack([-vplane[:,1], vplane[:,0], torch.zeros_like(vplane[:,0])], dim=1),
        dim=1)
    return vparam[:, :2] / vparam[:, [2]]

def np_vplane_2_vparam(vplane):
    """
    Numpy version of vplane_2_vparam
    Input:
        vplane: [..., 3]. The parameters <a, b, c> for line ax + by + c = 0

    Return:
        vparam: [..., 2]. The shortes vector from camera point to the plane
    """
    vparam = np.cross(
        vplane,
        np.stack([-vplane[...,1], vplane[...,0], np.zeros_like(vplane[...,0])], axis=-1),
        axis=-1)
    return vparam[..., :2] / vparam[..., [2]]

def vparam_2_vplane(vparam):
    """
    Inverse of vplane_2_vparam
    """
    d = torch.norm(vparam, p=2, dim=1, keepdim=True)
    a = vparam[:, [0]] / d
    b = vparam[:, [1]] / d
    neg_sign = (a < 0)
    a[neg_sign] = -a[neg_sign]
    b[neg_sign] = -b[neg_sign]
    c = -(a * vparam[:, [0]] + b * vparam[:, [1]])
    return torch.cat([a, b, c], 1)

def np_vparam_2_vplane(vparam):
    """
    Numpy version of vparam_2_vplane
    """
    d = np.linalg.norm(vparam, ord=2, axis=-1, keepdims=True)
    a = vparam[..., [0]] / d
    b = vparam[..., [1]] / d
    neg_sign = (a < 0)
    a[neg_sign] = -a[neg_sign]
    b[neg_sign] = -b[neg_sign]
    c = -(a * vparam[..., [0]] + b * vparam[..., [1]])
    vplane = np.concatenate([a, b, c], axis=-1)
    vplane[np.isnan(vplane)] = 0
    return vplane

def vparam_2_rad_d(vparam):
    """
    Input:
        vparam: [B, 2, H, W]. The shortes vector from camera point to the plane

    Return:
        rad:      [B, 1, H, W]. The rotation angle of each pixel of vparam
        radrel:  [B, 1, H, W]. Relative version of rad
        d:        [B, 1, H, W]. The lenght of each pixel of vparam
    """
    rad = torch.atan2(vparam[:, [1]], vparam[:, [0]])
    d = torch.norm(vparam, p=2, dim=1, keepdim=True)

    # Compute relative rad
    B, _, H_, W = vparam.shape
    us = u_grid(B, 1, H_, W).to(rad.device)
    radrel = rad - us
    radrel[radrel > np.pi] = radrel[radrel > np.pi] - 2*np.pi
    radrel[radrel < -np.pi] = radrel[radrel < -np.pi] + 2*np.pi
    return rad, d, radrel, us

def radrel_d_2_vparam(radrel, d):
    """
    Inverse of vparam_2_rad_d
    """
    # Compute relative rad
    B, _, H_, W = radrel.shape
    us = u_grid(B, 1, H_, W).to(radrel.device)
    rad = radrel + us
    x = torch.cos(rad) * d
    y = torch.sin(rad) * d
    return torch.cat([x, y], 1)

def depth_2_Q(depth):
    """
    Input:
        depth: [B, 1, H_, W]

    Return
        Q: [B, 3, H_, W] the 3d point cloud
    """
    B, _, H_, W = depth.shape
    us = u_grid(B, 1, H_, W).to(depth.device)
    vs = v_grid(B, 1, H_, W).to(depth.device)
    zs = depth * torch.sin(vs)
    xs = depth * torch.cos(vs) * torch.cos(us)
    ys = depth * torch.cos(vs) * torch.sin(us)
    return torch.cat([xs, ys, zs], 1)

def hplane_2_depth(hplane):
    """
    hplane: [B, 1, H_, W]
    depth: [B, 1, H_, W]
    """
    B, _, H_, W = hplane.shape
    vs = v_grid(B, 1, H_, W).to(hplane.device)
    depth = hplane / torch.sin(vs)
    return depth

def vparam_2_depth(vparam):
    """
    vparam: [B, 2, H_, W]
    depth: [B, 1, H_, W]
    """
    B, _, H_, W = vparam.shape
    us = u_grid(B, 1, H_, W).to(vparam.device)
    vs = v_grid(B, 1, H_, W).to(vparam.device)
    ray = torch.cat([
        torch.cos(vs) * torch.cos(us),
        torch.cos(vs) * torch.sin(us),
        torch.sin(vs),
    ], 1)  # B, 3, H_, W
    vparam = torch.cat([
        vparam, torch.zeros_like(vparam[:, [0]])
    ], 1)  # B, 3, H_, W

    A = (vparam * vparam).sum(1, keepdim=True)  # B, 1, H_, W
    B = (vparam * ray).sum(1, keepdim=True)     # B, 1, H_, W
    return A / B

def vplane_2_depth(vplane):
    """
    vplane: [B, 3, H_, W]
    depth: [B, 1, H_, W]
    """
    B, _, H_, W = vplane.shape
    us = u_grid(B, 1, H_, W).to(vplane.device)
    vs = v_grid(B, 1, H_, W).to(vplane.device)
    raycast = torch.cross(
        torch.cat([torch.cos(us), torch.sin(us), torch.ones_like(us)], 1),
        torch.FloatTensor([0, 0, 1]).reshape(1, 3, 1, 1).repeat(B, 1, H_, W).to(vplane.device),
        dim=1
    )
    pts2d = torch.cross(raycast, vplane, dim=1)
    pts2d = pts2d[:, :2] / (pts2d[:, [2]] + 1e-9)
    c = torch.norm(pts2d, p=2, dim=1, keepdim=True)
    depth = c / torch.cos(vs)
    return depth

def segmap_mean(segmap, param, zero_mask):
    '''
    segmap, param should in same shape
    '''
    shape = segmap.shape
    _, segmap = np.unique(segmap.reshape(-1), return_inverse=True)  # Reorder from 0 to N-1
    param = param.reshape(-1)
    param_sum = np.bincount(segmap, param)  # N bins
    param_cnt = np.bincount(segmap)         # N bins
    param_mean = param_sum / param_cnt      # N bins
    param = param_mean[segmap].reshape(shape)
    param[zero_mask] = 0
    return param


# Adapt from: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
def neg_lovasz_hinge(logits, labels):
    """
    Binary Lovasz hinge loss (negative positive invert version)
      logits: [N] logits at each prediction (between -\infty and +\infty)
      labels: [N] binary ground truth labels (0 or 1)
    """
    # Positive negative convert
    logits = -logits
    labels = 1 - labels.float()

    # Compute lovasz hinge as usual
    signs = 2 * labels - 1
    errors = (1 - signs * logits)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    n_pos = gt_sorted.sum()
    intersection = n_pos - gt_sorted.cumsum(0)
    union = n_pos + (1 - gt_sorted).cumsum(0)
    jaccard = 1 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

if __name__ == '__main__':
    pass
