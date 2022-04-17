import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpha_pred import AlphaModel
from .tex_gen import TexUNet
from .trajectory import *

import sys

sys.path.append("..")
import utils


class SpriteModel(nn.Module):
    """
    Full sprite model
    cfg loaded from src/confs/models
    """

    def __init__(self, dset, n_layers, cfg):
        super().__init__()

        self.dset = dset

        N, H, W = len(dset), dset.height, dset.width
        self.n_layers = n_layers

        ## initialize mask prediction
        args = cfg.alpha_pred
        self.alpha_pred = AlphaModel(n_layers, **dict(args))
        optims = [{"params": self.alpha_pred.parameters(), "lr": args.lr}]

        self.has_tex = cfg.use_tex
        if cfg.use_tex:
            ## initialize texture generator
            args = cfg.tex_gen
            TH, TW = args.scale_fac * H, args.scale_fac * W
            self.tex_shape = (TH, TW)
            self.tex_gen = TexUNet(n_layers, self.tex_shape, **dict(args))
            optims.append({"params": self.tex_gen.parameters(), "lr": args.lr})

            ## initialize motion model
            args = cfg.transform
            self.local = args.local
            self.active_local = False
            self.tforms = init_trajectory(
                dset,
                n_layers,
                active_local=self.active_local,
                **dict(args),
            )
            optims.append({"params": self.tforms.parameters(), "lr": args.lr})

            view_grid = utils.get_uv_grid(H, W, homo=True)  # (H, W, 3)
            self.register_buffer("view_grid", view_grid.view(1, 1, H, W, 3))

            cano_grid = utils.get_uv_grid(TH, TW, homo=True)  # (TH, TW, 3)
            self.register_buffer("cano_grid", cano_grid.view(1, 1, TH, TW, 3))

        self.optim = torch.optim.Adam(optims)
        self.skip_keys = ["alpha", "pred"]

    def forward(
        self,
        batch,
        quantile=0.8,
        ret_tex=True,
        ret_tform=True,
        vis=False,
        ret_inputs=False,
    ):
        out = {}

        alpha_dict = self.alpha_pred(batch["rgb"])
        out.update(alpha_dict)
        masks = out["masks"]
        B, M, _, H, W = masks.shape

        ret_tex = ret_tex and self.has_tex
        ret_tform = (ret_tex or ret_tform) and self.has_tex

        if ret_tform:
            ## get the coordinates from view to canonical
            tform_dict = self.get_view2cano_coords(batch["idx"])
            out.update(tform_dict)

        if ret_tex:
            ## get the canonical textures and warped appearances
            ## texs (M, 3, H, W) and apprs (B, M, 3, H, W)
            ## from coords (B, M, H, W, 2)
            tex_dict = self.tex_gen(out["coords"], vis=vis)
            out.update(tex_dict)

            ## composite layers
            out["recons"] = (masks * out["apprs"]).sum(dim=1)  # (B, 3, H, W)
            out["layers"] = utils.composite_rgba_checkers(masks, out["apprs"])

        if vis:
            out["masks_vis"] = utils.composite_rgba_checkers(masks, 1)
            if ret_tform:
                out["cano_vis"] = self.get_cano2view_vis(
                    batch["idx"], masks, out.get("texs", None)
                )
                out["view_vis"] = self.get_view2cano_vis(out["coords"], masks)

        if ret_inputs:
            out["rgb"] = batch["rgb"]
            out["idx"] = batch["idx"]
            out["flow"] = utils.flow_to_image(batch["fwd"][1])
            out["flow_groups"] = utils.composite_rgba_checkers(
                masks, out["flow"][:, None]
            )

        return out

    def get_view2cano_coords(self, idx):
        B, M = len(idx), self.n_layers
        view_grid = self.view_grid.expand(B, M, -1, -1, -1)  # (B, M, H, W, 3)
        return {
            "coords": self.tforms(idx, view_grid),
            "view_grid": view_grid,
        }  # (B, M, H, W, 2)

    def get_view2cano_vis(self, coords, masks=None, nrows=16):
        """
        :param coords (B, M, H, W, 2)
        :param masks (optional) (B, M, 1, H, W)
        :param nrows (optional) (int)
        """
        B, M = coords.shape[:2]
        TH, TW = self.tex_shape
        device = coords.device
        cano_grid = utils.get_rainbow_checkerboard(
            TH, TW, nrows, device=device
        )  # (3, H, W)
        cano_grid = cano_grid[None, None].repeat(B, M, 1, 1, 1)
        view_grid = utils.resample_batch(cano_grid, coords, align_corners=False)
        if masks is None:
            masks = (view_grid != 0).float()

        view_grid = utils.composite_rgba_checkers(masks, view_grid)
        return view_grid

    def get_cano2view_vis(self, idx, masks, texs=None, fac=0.3, nrows=16):
        """
        :param idx (B)
        :param masks (B, M, 1, H, W)
        :param texs (1, M, 3, H, W)
        """
        B, M, _, H, W = masks.shape
        cano_grid = self.cano_grid.expand(B, M, -1, -1, -1)  # (B, M, TH, TW, 3)
        cano2view = self.tforms.get_cano2view(idx)  # (B, M, 3, 3)
        view_coords = utils.apply_homography_xy1(
            cano2view, cano_grid
        )  # (B, M, H, W, 2)
        view = utils.get_rainbow_checkerboard(
            H, W, nrows, device=idx.device
        )  # (3, H, W)
        view = view[None, None] * masks  # (1, 1, 3, H, W) * (B, M, 1, H, W)
        cano_frames = torch.stack(
            [
                F.grid_sample(view[:, i], view_coords[:, i], align_corners=False)
                for i in range(M)
            ],
            dim=1,
        )  # (B, M, 3, H, W)
        if texs is None:
            return cano_frames

        return fac * cano_frames + (1 - fac) * texs

    def init_planar_motion(self, masks):
        if self.has_tex:
            fwd_set = self.dset.get_set("fwd")
            trans, scale, uv_range, ok = estimate_displacements(fwd_set, masks)
            print("DISPLACEMENTS", ok)
            print("SCALE", scale[0])
            print("TRANS", trans[0])
            print("UV_RANGE", uv_range)
            self.tforms.update_trans(trans)
            self.tforms.update_scale(scale)
            return ok
        return False

    def init_local_motion(self):
        if self.has_tex and self.local:
            self.active_local = True
            self.tforms.init_local_field()
