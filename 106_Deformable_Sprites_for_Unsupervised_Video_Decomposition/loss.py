import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import data


class ReconLoss(nn.Module):
    """
    Mixed L2 and Laplacian loss
    """

    def __init__(
        self,
        weight,
        lap_ratio=1e-3,
        norm=1,
        n_levels=3,
        ksize=3,
        sigma=1,
        detach_mask=True,
    ):
        super().__init__()
        self.weight = weight
        self.lap_ratio = lap_ratio
        self.abs_fnc = torch.abs if norm == 1 else torch.square
        self.n_levels = n_levels
        self.ksize = ksize
        self.sigma = sigma
        self.detach_mask = detach_mask

    def forward(self, batch_in, batch_out, split=False):
        rgb = batch_in["rgb"]
        if self.weight <= 0:
            return None
        if self.detach_mask:
            appr = batch_out["apprs"]
            mask = batch_out["masks"].detach()
            recon = (mask * appr).sum(dim=1)
        else:
            recon = batch_out["recons"]

        l2_err = self.abs_fnc(recon - rgb).mean()

        recon_lp = utils.get_laplacian_pyr(recon, self.n_levels, self.ksize, self.sigma)
        target_lp = utils.get_laplacian_pyr(rgb, self.n_levels, self.ksize, self.sigma)
        H, W = recon.shape[-2:]
        lap_err = sum(
            [
                torch.abs(recon_lp[i] - target_lp[i]).sum() * (4 ** i)
                for i in range(self.n_levels)
            ]
        ) / (H * W)

        return self.weight * (l2_err + self.lap_ratio * lap_err)


class EpipolarLoss(nn.Module):
    """
    Penalize background layer's pixels with high sampson error,
    lightly penalizes the foreground layers' pixels with low sampson error
    """

    def __init__(self, weight, neg_ratio=2e-3, clip=10.0):
        super().__init__()
        self.weight = weight
        self.neg_ratio = neg_ratio
        self.clip = clip

    def whiten_distance(self, err):
        e_max = err.max()
        #         e_max = torch.clamp_min(e_max, self.clip)
        err = torch.clamp_max(err, e_max) / e_max
        return err

    def forward(self, batch_in, batch_out):
        """
        we pre-compute sampson error with a fundamental matrix computed with LMeDS,
        and threshold the sampson error with the median
        """
        ok, err, _ = batch_in["epi"]  # (B, H, W)
        if ok.sum() < 1:
            return None

        masks = batch_out["masks"][ok]  # (B, M, 1, H, W)
        err = self.whiten_distance(err[ok])
        bg_mask = masks[:, -1, 0]  # (B, H, W)
        loss = bg_mask * err + self.neg_ratio * (1 - bg_mask) * (1 - err)
        return self.weight * loss.mean()

    def vis(self, batch_in, batch_out):
        ok, err, _ = batch_in["epi"]  # (B, H, W)
        err = self.whiten_distance(err[ok])
        return {"epi": err[:, None, None]}


def get_stats(X, norm=2):
    """
    :param X (N, C, H, W)
    :returns mean (1, C, 1, 1), scale (1)
    """
    mean = X.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


class FlowGroupingLoss(nn.Module):
    def __init__(
        self,
        weight,
        norm=1,
        sep_fac=0.1,
        bg_fac=2.0,
        detach_mean=True,
    ):
        super().__init__()
        self.weight = weight
        self.norm_fnc = torch.abs if norm == 1 else torch.square
        self.detach_mean = detach_mean
        self.sep_fac = sep_fac
        self.bg_fac = bg_fac

    def forward(self, batch_in, batch_out, split=False):
        """
        :param masks (*, 1, H, W)
        :param src (*, C, H, W)
        """
        ok, flow = batch_in["fwd"]

        if ok.sum() < 1:
            print("NO FLOWS")
            return None

        masks = batch_out["masks"][ok]

        B, M, _, H, W = masks.shape
        device = masks.device

        flow = flow[ok]  # (B, 2, H, W)
        f_mean, f_std = get_stats(flow)
        flow = ((flow - f_mean) / f_std).unsqueeze(1)

        with torch.no_grad():
            mass = masks.sum(dim=(-1, -2), keepdim=True) + 1e-6
            mean = (masks * flow).sum(dim=(-1, -2), keepdim=True) / mass

        dists = self.norm_fnc(flow - mean).sum(dim=2, keepdim=True)

        fac = torch.cat(
            [
                torch.ones(M - 1, device=device),
                self.bg_fac * torch.ones(1, device=device),
            ]
        )
        rand = torch.cat(
            [
                torch.zeros(1, device=device),
                self.sep_fac * (torch.rand(M - 1, device=device) + 1),
            ]
        )
        masks = masks * (fac + rand).view(1, -1, 1, 1, 1)

        wdists = masks * dists
        return self.weight * wdists.mean()


class FlowWarpLoss(nn.Module):
    def __init__(
        self,
        weight,
        tforms,
        gap,
        src_name="fwd",
        norm=1,
        unscaled=False,
        detach_mask=True,
    ):
        """
        Loss that supervises the view->cano transform for each frame
        For a point A in view, T1(A) takes A -> A' in cano
        FLOW_12(A) takes A -> B in view, T2(B) takes B -> B' in cano
        A' and B' should be the same point in cano

        Minimizes distance between A' and B' in cano

        :param gap (int) the spacing between flow pairs
        :param norm (int, optional) the norm to use for distance
        """

        super().__init__()
        print("Initializing flow warp loss with {} and {}".format(src_name, gap))
        self.weight = weight
        self.tforms = tforms
        self.gap = gap
        self.src_name = src_name
        self.norm_fnc = torch.abs if norm == 1 else torch.square
        self.detach_mask = detach_mask
        self.unscaled = unscaled

    def forward(self, batch_in, batch_out, split=False, no_reduce=False):
        gap = self.gap
        idx = batch_in["idx"]

        if self.weight <= 0 or len(idx) < abs(gap):
            return None

        # flow (B, 2, H, W) -> (B, H, W, 2)
        ok, flow = batch_in[self.src_name]
        flow = flow.permute(0, 2, 3, 1)

        masks = batch_out["masks"]  # (B, M, 1, H, W)
        coords = batch_out["coords"]  # (B, M, H, W, 2)
        grid = batch_out["view_grid"]  # (B, M, H, W, 3)

        B, M, _, H, W = masks.shape

        if self.detach_mask:  # primarily fitting the transform
            masks = masks.detach()

        masks = masks.view(B, M, H, W)

        if gap > 0:
            # 0 ... B-1-gap
            W1 = masks[:-gap]
            V1 = ok[:-gap]
            I1 = idx[:-gap]
            P1 = coords[:-gap]
            G1 = grid[:-gap]
            F12 = flow[:-gap]
            # gap ... B-1
            I2 = idx[gap:]
            V2 = ok[gap:]
        else:
            # gap ... B-1
            W1 = masks[-gap:]
            V1 = ok[-gap:]
            I1 = idx[-gap:]
            P1 = coords[-gap:]
            G1 = grid[-gap:]
            F12 = flow[-gap:]
            # 0 ... B-1-gap
            I2 = idx[:gap]
            V2 = ok[:gap]

        valid = V1 & V2
        I1 = I1[valid]
        W1 = W1[valid]
        P1 = P1[valid]
        G1 = G1[valid]
        F12 = F12[valid]
        I2 = I2[valid]

        F12 = torch.cat([F12, torch.zeros_like(F12[..., -1:])], dim=-1)
        G2 = G1 + F12[:, None]
        P2 = self.tforms(I2, grid=G2)

        # rescale to pixel coordinates (0 - W, 0 - H)
        scale_fac = (W + H) / 4

        if self.unscaled:
            s1 = self.tforms.get_scale(I1).view(-1, M, 1, 1, 1)
            s2 = self.tforms.get_scale(I2).view(-1, M, 1, 1, 1)
            s1 = torch.cat([s1[:, :-1], torch.ones_like(s1[:, -1:])], dim=1)
            s2 = torch.cat([s2[:, :-1], torch.ones_like(s2[:, -1:])], dim=1)
            diffs = scale_fac * (P1 - P2) / (s1 + s2 + 1e-5)
        else:
            diffs = scale_fac * (P2 - P1)

        wdists = W1 * self.norm_fnc(diffs).sum(dim=-1)
        return self.weight * wdists.mean()


class MaskWarpLoss(nn.Module):
    def __init__(self, weight, gap, norm=1):
        super().__init__()
        assert norm == 1 or norm == 2 or norm == "xent"
        self.gap = gap
        self.weight = weight
        self.norm = norm

    def forward(self, batch_in, batch_out, split=False):
        """
        We consider point A in frame 1. Flow takes A to B in frame 2: FLOW_12(A) -> B.
        The mask value of point A in 1 should be the same as the mask value of point B in 2,
        unless point A is occluded.
        :param masks (B, M, 1, H, W)
        :param flow (B, 2, H, W)
        :param occ_map (B, 1, H, W)
        return (B-gap, M, H, W) distance between the corresponding masks
        """
        gap = self.gap
        masks = batch_out["masks"]  # (B, M, 1, H, W)

        if len(masks) < abs(self.gap):  # not enough in batch
            return torch.zeros(1, dtype=torch.float32, device=masks.device)

        if gap > 0:
            ok, flow = batch_in["fwd"]  # (B, 2, H, W)
            occ_map = batch_in["occ"][0]
            ok = ok[:-gap]
            F12 = flow[:-gap].permute(0, 2, 3, 1)  # 0 ... B-1-gap
            O12 = occ_map[:-gap]
            M1 = masks[:-gap, :, 0, ...]  # 0 ... B-1-gap
            M2 = masks[gap:, :, 0, ...]  # gap ... B-1
        else:
            ok, flow = batch_in["bck"]
            occ_map = batch_in["disocc"][0]
            ok = ok[-gap:]
            F12 = flow[-gap:].permute(0, 2, 3, 1)  # gap ... B-1
            O12 = occ_map[-gap:]
            M1 = masks[-gap:, :, 0, ...]  # gap ... B-1
            M2 = masks[:gap, :, 0, ...]  # 0 ... B-1-gap

        M1, M2 = M1[ok], M2[ok]
        F12, O12 = F12[ok], O12[ok]

        ## mask 1 resampled from mask 2
        W1 = utils.inverse_flow_warp(M2, F12, O12)

        if self.norm == 1:
            dist = (~O12) * torch.abs(W1 - M1)
        elif self.norm == 2:
            dist = (~O12) * torch.square(W1 - M1)
        elif self.norm == "xent":
            W1 = W1.detach()  # (B, M, H, W)
            dist = -W1 * torch.log(M1 + 1e-8) - (1 - W1) * torch.log(1 - M1 + 1e-8)
        else:
            raise NotImplementedError

        return self.weight * dist.mean()


class ContrastiveTexLoss(nn.Module):
    def __init__(self, weight, thresh=0.25, use_mask=False, detach_mask=False):
        super().__init__()
        self.weight = weight
        self.thresh = thresh
        self.use_mask = use_mask
        self._detach_mask = detach_mask

    def detach_mask(self):
        self._detach_mask = True

    def attach_mask(self):
        self.use_mask = True
        self._detach_mask = False

    def forward(self, batch_in, batch_out, split=False):
        apprs = batch_out["apprs"]  # (B, M, _, H, W)
        B, M = apprs.shape[:2]

        ## compute the similarity between each pair of layers
        sim = (apprs.unsqueeze(2) * apprs.unsqueeze(1)).sum(dim=3)  # (B, M, M, H, W)

        ## zero out the diagonals (similarity with itself)
        idcs = torch.arange(M, device=apprs.device)  # (M)
        sim[:, idcs, idcs] = 0

        if self.use_mask:
            ## for every layer, apply its mask on the other layers
            ## we don't want the other layer appearances to be similar
            ## in the regions that should be explained only by this layer
            masks = batch_out["masks"]  # (B, M, 1, H, W)
            if self._detach_mask:
                masks = masks.detach()
            masks = masks > self.thresh
            sim = masks * sim

        return self.weight * sim.mean()


def compute_losses(loss_fncs, batch_in, batch_out):
    loss_dict = {}
    for name, fnc in loss_fncs.items():
        if fnc.weight <= 0:
            continue
        loss = fnc(batch_in, batch_out)
        if loss is None:
            continue
        loss_dict[name] = loss
    return loss_dict


def get_loss_grad(batch_in, batch_out, loss_fncs, var_name, loss_name=None):
    """
    get the gradient of selected losses wrt to selected variables
    Which losses and which variable are specified with a list of tuples, grad_pairs
    """
    ## NOTE: need to re-render to re-populate computational graph
    ## in future maybe can also retain graph
    var = batch_out[var_name]
    *dims, C, H, W = var.shape

    var.retain_grad()
    sel_fncs = {loss_name: loss_fncs[loss_name]} if loss_name is not None else loss_fncs
    loss_dict = compute_losses(sel_fncs, batch_in, batch_out)
    if len(loss_dict) < 1:
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    try:
        sum(loss_dict.values()).backward()
    except:
        pass

    if var.grad is None:
        print("requested grad for {} wrt {} not available".format(loss_name, var_name))
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    return utils.get_sign_image(var.grad.detach())
