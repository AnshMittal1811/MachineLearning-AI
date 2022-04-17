from functools import partial

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import get_nl
from .planar import PlanarMotion

import sys

sys.path.append("..")
import utils


def init_trajectory(dset, n_layers, local=False, **kwargs):
    N, H, W = len(dset), dset.height, dset.width
    if local:
        return BSplineTrajectory(N, n_layers, (H, W), **kwargs)
    return PlanarTrajectory(n_total, n_layers, **kwargs)


def estimate_displacements(
    flow_dset, masks, init_scale=0.8, thresh=0.5, pad=0.1, fg_pad=0.2, bg_scale=1
):
    """
    roughly estimate how much the scene is displaced per frame
    :params masks (N, M, 1, H, W)
    :returns trans (N, M, 2), scale (N, M, 2), uv_range (M, 2)
    """
    assert len(flow_dset) == len(masks)
    device = masks.device
    N, M = masks.shape[:2]
    with torch.no_grad():
        sel = binarize_masks(masks[:, :, 0], thresh)  # (N, M, H, W)

    imageio.mimwrite("init_masks.gif", (255 * sel[:, 0].cpu()).byte())

    # get the mean flows for each layer for all frames
    # flow element is tuple (valid, flow)
    flows = [flow_dset[i][1].to(device) for i in range(N)]
    flow_vecs = [
        flows[i][:, sel[i, j]] for i in range(N) for j in range(M)
    ]  # B*M list (2, -1)
    med_flow = reduce_tensors(flow_vecs, torch.mean, dim=-1).reshape(N, M, 2)
    med_flow = torch.where(med_flow.isnan(), torch.zeros_like(med_flow), med_flow)

    # estimate the bboxes of each layer except background
    bb_min, bb_max, ok = compute_bboxes(sel[:, :-1])
    bb_min = torch.cat(
        [bb_min - fg_pad, -bg_scale * torch.ones(N, 1, 2, device=device)], dim=1
    )
    bb_max = torch.cat(
        [bb_max + fg_pad, bg_scale * torch.ones(N, 1, 2, device=device)], dim=1
    )

    return (*compute_scale_trans(med_flow, bb_min, bb_max, init_scale, pad), ok)


def compute_bboxes(sel, default=0.5):
    """
    :param sel (N, M, H, W)
    :returns bb_min (N, M, 2), bb_max (N, M, 2), ok (N, M)
    """
    N, M, H, W = sel.shape
    uv = utils.get_uv_grid(H, W, device=sel.device).permute(2, 0, 1)  # (2, H, W)
    bb_min = -0.5 * torch.ones(N, M, 2, device=sel.device)
    bb_max = 0.5 * torch.ones(N, M, 2, device=sel.device)

    # manually fill in the non-empty slots (min is undefined for empty)
    ok = (
        sel.sum(dim=(-1, -2)) > 0
    )  # (N, M) which layers of which frames have non-empty bboxes
    ii, jj = torch.where(ok)  # each (T ~ N*M)
    uv_vecs = [uv[:, sel[i, j]] for i, j in zip(ii, jj)]
    if len(uv_vecs) > 0:
        bb_min[ii, jj] = reduce_tensors(uv_vecs, torch.amin, dim=-1)  # (T, 2)
        bb_max[ii, jj] = reduce_tensors(uv_vecs, torch.amax, dim=-1)  # (T, 2)
    return bb_min, bb_max, len(uv_vecs) > 0


def compute_scale_trans(med_flow, bb_min, bb_max, init_scale=0.9, pad=0.2):
    """
    :param med_flow (N, M, 2)
    :param bb_min (N, M, 2)
    :param bb_max (N, M, 2)
    :returns trans (N, M, 2), scale (N, M, 2), uv_range (M, 2)
    """
    N, M, _ = med_flow.shape
    disp = -torch.cumsum(med_flow, dim=0)  # (N, M, 2)
    disp = torch.cat([torch.zeros_like(disp[:1]), disp[:-1]], dim=0)

    # align bboxes using estimated displacement from first frame
    align_min = bb_min + disp
    align_max = bb_max + disp
    min_coord = torch.quantile(align_min, 0.2, dim=0, keepdim=True) - pad  # (1, M, 2)
    max_coord = torch.quantile(align_max, 0.8, dim=0, keepdim=True) + pad  # (1, M, 2)
    uv_range = torch.abs(max_coord - min_coord)  # (1, M, 2)

    scale = init_scale * 2 / uv_range.repeat(N, 1, 1)  # (N, M, 2)
    trans = scale * (disp - min_coord) - init_scale  # (N, M, 2)

    return trans, scale, uv_range


def binarize_masks(masks, thresh=0.5):
    """
    :param masks (B, M, H, W)
    :return sel (B, M, H, W)
    """
    sel = (masks > thresh).float()
    sel = F.max_pool2d(1 - sel, 3, 1, 1)
    sel = 1 - F.max_pool2d(sel, 3, 1, 1)
    return sel.bool()


def reduce_tensors(tensor_list, fnc, dim=-1):
    """
    reduce tensors in a list with fnc along dim, then stack
    :param tensor_list (list of N (*, d) tensors)
    :returns (N, *) tensor
    """
    if len(tensor_list) < 1:
        return torch.tensor([])
    return torch.stack([fnc(t, dim=dim) for t in tensor_list], dim=0)


class PlanarTrajectory(PlanarMotion):
    """
    Planar motion model that interpolates motion parameters between frames
    (enforces temporal smoothness)
    """

    def __init__(
        self, n_total, n_layers, t_step=1, degree=2, scale=None, trans=None, **kwargs
    ):
        nk_t = (n_total - 1) // t_step + 1
        tk_range = int(t_step * nk_t)

        self.n_total = n_total
        self.t_step = t_step
        self.nk_t = nk_t
        self.tk_range = tk_range
        self.degree = degree

        if trans is not None:
            trans = trans[::t_step]
        if scale is not None:
            scale = scale[::t_step]

        ## keep an explicit motion field for the knots
        super().__init__(nk_t, n_layers, scale=scale, trans=trans, **kwargs)

    def get_theta(self, idx):
        """
        :param idx (B)
        """
        if self.nk_t == self.tk_range:
            return super().get_theta(idx)

        N, M, *dims = self.theta.shape
        knots = self.theta.transpose(0, 1).view(M, N, -1)  # (M, nk_t, -1)
        # knots[0] <-- 0, knots[n] <-- N - 1
        idx = idx / (self.tk_range - 1) * (self.nk_t - 1) - 0.5
        positions = idx.view(1, -1).expand(M, -1)  # (M, B)
        sel_theta = utils.bspline.interpolate_1d(
            knots, positions, self.degree
        )  # (M, B, -1)
        sel_theta = sel_theta.transpose(0, 1).view(-1, M, *dims)
        return sel_theta

    def update_scale(self, scale):
        """
        :param scale (N, M, 2)
        """
        if len(scale) == self.n_total:
            scale = scale[:: self.t_step]
        super().update_scale(scale)

    def update_trans(self, trans):
        """
        :param trans (N, M, 2)
        """
        if len(trans) == self.n_total:
            trans = trans[:: self.t_step]
        super().update_trans(trans)


class BSplineTrajectory(PlanarTrajectory):
    """
    Planar motion + 2D splines
    """

    def __init__(
        self,
        n_total,
        n_layers,
        out_shape,
        t_step=1,
        xy_step=8,
        final_nl="tanh",
        active_local=True,
        bg_local=True,
        max_step=0.1,
        **kwargs
    ):
        super().__init__(n_total, n_layers, t_step=t_step, **kwargs)

        H, W = out_shape
        nk_x, nk_y = W // xy_step, H // xy_step

        self.active_local = active_local
        self.bg_local = bg_local

        nk_layers = n_layers if bg_local else n_layers - 1
        knots = torch.zeros(nk_layers, self.nk_t, nk_y, nk_x, 2)
        self.register_parameter("knots_3d", nn.Parameter(knots, requires_grad=True))
        print(
            "Initialized BSpline motion with {} knots".format((self.nk_t, nk_y, nk_x))
        )
        print("knots_3d.shape:", knots.shape)

        self.final_nl = get_nl(final_nl)
        max_step = torch.cat(
            [torch.ones(n_layers - 1) * max_step, torch.ones(1) * 0.5 * max_step]
        )
        self.register_buffer("max_step", max_step.view(1, n_layers, 1, 1, 1))

    def get_rigid_transform(self, idx, grid):
        return super().forward(idx, grid)

    def init_local_field(self):
        self.active_local = True
        print("MOTION FIELD NOW LOCAL")

    def get_knots(self):
        knots = self.knots_3d.transpose(0, 1)  # (N, M, h, w, 2)
        return knots.permute(0, 1, 4, 2, 3)  # (N, M, 2, h, w)

    def get_knots_xy(self, idx):
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        if nk_t == self.tk_range:
            knots_xy = self.knots_3d[:, idx]  # (M, B, nk_y, nk_x, D)
            return knots_xy.view(-1, nk_y, nk_x, D)

        ## interpolate in time, independently in every dimension
        B = idx.shape[0]
        knots_t = self.knots_3d.view(M, nk_t, -1)  # (M, nk_t, nk_y*nk_x*D)
        # need to rescale the query points in terms of number of knots
        idx = idx / (self.tk_range - 1) * (nk_t - 1) - 0.5
        positions_t = idx.view(1, -1).expand(M, -1)  # (M, B)

        ## query the 2d control points in time
        knots_xy = utils.bspline.interpolate_1d(
            knots_t, positions_t, self.degree
        )  # (M, B, -1)
        knots_xy = knots_xy.view(-1, nk_y, nk_x, D)  # (M*B, nk_y, nk_x, 2)
        return knots_xy

    def get_local_offsets(self, idx, grid):
        """
        :param idx (B)
        :param grid (B, M, H, W, 3)
        """
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        B = idx.shape[0]
        H, W, _ = grid.shape[-3:]

        knots_xy = self.get_knots_xy(idx)

        ## set up the query grid
        grid = grid[:, :M, ..., :2]  # (B, M, H, W, 2)
        grid = grid.transpose(0, 1).reshape(-1, H, W, 2)

        # rescale x and y from [-1, 1] --> [-0.5, nk - 1.5]
        fac = torch.tensor([(nk_x - 1) / 2, (nk_y - 1) / 2], device=grid.device).view(
            1, 1, 1, 2
        )
        query_grid = (grid + 1) * fac - 0.5
        offsets = utils.bspline.interpolate_2d(
            knots_xy, query_grid, self.degree
        )  # (M*B, H, W, 2)
        offsets = offsets.view(M, B, H, W, 2).transpose(0, 1)  # (B, M, H, W, 2)

        if not self.bg_local:
            # we don't keep 2D spline for background, add zeros
            offsets = torch.cat([offsets, torch.zeros_like(offsets[:, :1])], dim=1)

        offsets = self.final_nl(offsets) * self.max_step
        return offsets

    def forward(self, idx, grid):
        t_rigid = self.get_rigid_transform(idx, grid)  # (B, M, H, W, 2)
        if self.active_local:
            t_local = self.get_local_offsets(idx, grid)  # (B, M, H, W, 2)
            #             print(t_local.square().sum(dim=-1).mean())
            return t_rigid + t_local
        return t_rigid
