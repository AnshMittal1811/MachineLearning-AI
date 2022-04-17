import os
import glob

from functools import partial

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import data
import models
import utils
from loss import *
from helper import *


DEVICE = torch.device("cuda")


@hydra.main(config_path="confs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dset = get_dataset(cfg.data)
    N, H, W = len(dset), dset.height, dset.width
    can_preload = N < 200 and cfg.data.scale < 0.5
    preloaded = cfg.preload and can_preload

    loader = data.get_random_ordered_batch_loader(
        dset,
        cfg.batch_size,
        preloaded,
    )
    val_loader = data.get_ordered_loader(
        dset,
        cfg.batch_size,
        preloaded,
    )

    model = models.SpriteModel(dset, cfg.n_layers, cfg.model)
    model.to(DEVICE)

    ## determines logging dir in hydra config
    log_dir = os.getcwd()
    writer = SummaryWriter(log_dir=log_dir)
    print("SAVING OUTPUT TO:", log_dir)

    if preloaded:
        dset.set_device(DEVICE)

    # optimize the model in phases
    flow_gap = cfg.data.flow_gap
    cfg = update_config(cfg, loader)
    save_args = dict(
        writer=writer,
        vis_every=cfg.vis_every,
        val_every=cfg.val_every,
        vis_grad=cfg.vis_grad,
        batch_size=cfg.batch_size,
    )
    loss_fncs = {
        "f_warp": MaskWarpLoss(cfg.w_warp, flow_gap),
        "b_warp": MaskWarpLoss(cfg.w_warp, -flow_gap),
    }

    opt_infer_helper = partial(
        opt_infer_step,
        loader=loader,
        val_loader=val_loader,
        model=model,
        loss_fncs=loss_fncs,
        **save_args,
    )

    # warmstart the masks
    label = "masks"
    model_kwargs = dict(ret_tex=False, ret_tform=False)
    if cfg.epochs_per_phase["epi"] > 0:
        dset.get_set("epi").save_to(log_dir)
        loss_fncs["epi"] = EpipolarLoss(cfg.w_epi)

    if cfg.epochs_per_phase["kmeans"] > 0:
        loss_fncs["kmeans"] = FlowGroupingLoss(cfg.w_kmeans)

    n_epochs = cfg.epochs_per_phase["epi"] + cfg.epochs_per_phase["kmeans"]
    step_ct, val_dict = opt_infer_helper(
        n_epochs, model_kwargs=model_kwargs, label=label
    )

    if not model.has_tex:
        return

    # warmstart planar transforms
    label = "planar"
    n_epochs = cfg.epochs_per_phase[label]
    loss_fncs["tform"] = FlowWarpLoss(cfg.w_tform, model.tforms, flow_gap)
    if "kmeans" in loss_fncs:
        loss_fncs["kmeans"].weight = 0.01 * cfg.w_kmeans
    loss_fncs["recon"] = ReconLoss(
        cfg.w_recon, cfg.lap_ratio, cfg.l_recon, cfg.lap_levels
    )
    loss_fncs["contr"] = ContrastiveTexLoss(cfg.w_contr)

    ok = model.init_planar_motion(val_dict["masks"].to(DEVICE))
    if not ok:
        # warmstart before estimating scale of textures
        n_warm = n_epochs // 2
        loss_fncs["tform"].detach_mask = False
        step_ct, val_dict = opt_infer_helper(
            n_warm, start=step_ct, label=label
        )
        # re-init scale of textures with rough planar motion
        model.init_planar_motion(val_dict["masks"].to(DEVICE))

    step_ct, val_dict = opt_infer_helper(
        n_epochs, start=step_ct, label=label
    )

    # add deformations
    label = "deform"
    model.init_local_motion()
    loss_fncs["tform"].unscaled = True

    n_epochs = cfg.epochs_per_phase[label]
    step_ct, val_dict = opt_infer_helper(
        n_epochs, start=step_ct, label=label
    )

    # refine masks with gradients through recon loss
    # very easy to cheat with these gradients, not recommended
    label = "refine"
    loss_fncs["recon"].detach_mask = False
    n_epochs = cfg.epochs_per_phase[label]
    if n_epochs < 1:
        return

    step_ct, val_dict = opt_infer_helper(
        n_epochs, start=step_ct, label=label
    )


if __name__ == "__main__":
    main()
