import os
import glob
from functools import partial

import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm

import data
import models
import utils
from loss import *


DEVICE = torch.device("cuda")


def get_dataset(args):
    rgb_dir, fwd_dir, bck_dir, gt_dir = data.get_data_dirs(
        args.type, args.root, args.seq, args.flow_gap, args.res
    )
    required_dirs = [rgb_dir, fwd_dir, bck_dir]
    assert all(d is not None for d in required_dirs), required_dirs

    rgb_dset = data.RGBDataset(rgb_dir, scale=args.scale)
    fwd_dset = data.FlowDataset(fwd_dir, args.flow_gap, rgb_dset=rgb_dset)
    bck_dset = data.FlowDataset(bck_dir, -args.flow_gap, rgb_dset=rgb_dset)
    epi_dset = data.EpipolarDataset(fwd_dset)
    occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)
    disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset)

    dsets = {
        "rgb": rgb_dset,
        "fwd": fwd_dset,
        "bck": bck_dset,
        "epi": epi_dset,
        "occ": occ_dset,
        "disocc": disocc_dset,
    }
    if gt_dir is not None:
        dsets["gt"] = data.MaskDataset(gt_dir, rgb_dset=rgb_dset)

    return data.CompositeDataset(dsets)


def optimize_model(
    n_epochs,
    loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    label=None,
    writer=None,
    vis_every=0,
    vis_grad=False,
    **kwargs,
):
    step_ct = start
    out_name = None if label is None else f"tr_{label}"
    save_vis = vis_every > 0 and out_name is not None
    for _ in tqdm(range(n_epochs)):
        for batch in loader:
            model.optim.zero_grad()
            batch = utils.move_to(batch, DEVICE)
            out_dict = model(batch, **model_kwargs)
            loss_dict = compute_losses(loss_fncs, batch, out_dict)
            step_ct += len(batch["idx"])

            if len(loss_dict) < 1:
                continue

            sum(loss_dict.values()).backward()
            model.optim.step()

            if writer is not None:
                for name, loss in loss_dict.items():
                    writer.add_scalar(f"loss/{name}", loss.item(), step_ct)

            if save_vis and step_ct % vis_every < len(batch["idx"]):
                save_dir = "{:08d}_{}".format(step_ct, out_name)
                if vis_grad:
                    out_dict = get_vis_batch(
                        batch, model, model_kwargs, loss_fncs, vis_grad
                    )
                utils.save_vis_dict(save_dir, out_dict)

    return step_ct


def get_vis_batch(
    batch, model, model_kwargs={}, loss_fncs={}, vis_grad=False, **kwargs
):
    batch = utils.move_to(batch, DEVICE)
    out_dict = model(batch, vis=True, ret_inputs=True, **model_kwargs)

    # save mask gradients if loss functions
    if vis_grad and len(loss_fncs) > 0:
        grad_img, grad_max = get_loss_grad(batch, out_dict, loss_fncs, "pred")
        out_dict["pred_grad"] = grad_img

    return out_dict


def infer_model(
    step_ct,
    loader,
    model,
    model_kwargs={},
    loss_fncs={},
    label=None,
    skip_keys=[],
):
    """
    run the model on all data points
    """
    out_name = None if label is None else f"{step_ct:08d}_val_{label}"
    print("val step {:08d} saving to {}".format(step_ct, out_name))
    out_dicts = []
    for batch in loader:
        batch = utils.move_to(batch, DEVICE)
        with torch.no_grad():
            out_dict = get_vis_batch(batch, model, model_kwargs)
            out_dict = compute_multiple_iou(batch, out_dict)

        out_dicts.append(
            {k: v.detach().cpu() for k, v in out_dict.items() if k not in skip_keys}
        )

    out_dict = utils.cat_tensor_dicts(out_dicts)
    if out_name is not None:
        if "texs" in out_dict:
            out_dict["texs"] = out_dict["texs"][:1]  # (n_batches, *) -> (1, *)
        utils.save_vis_dict(out_name, out_dict)
        # save the per-frame texture coords
        if "coords" in out_dict:
            torch.save(out_dict["coords"], f"{out_name}/coords.pth")
        save_metric(out_name, out_dict, "ious")
    return out_dict, out_name


def opt_infer_step(
    n_epochs,
    loader,
    val_loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    val_every=0,
    batch_size=16,
    label="model",
    ckpt=None,
    **kwargs,
):
    """
    optimizes model for n_epochs, then saves a checkpoint and validation visualizations
    """
    if ckpt is None:
        ckpt = "{}_latest_ckpt.pth".format(label)

    step = start
    steps_total = n_epochs * len(loader) * batch_size
    val_epochs = max(1, steps_total // val_every)
    n_epochs_per_val = max(1, n_epochs // val_epochs)
    print(f"running {val_epochs} train/val steps with {n_epochs_per_val} epochs each.")

    for _ in range(val_epochs):
        step = optimize_model(
            n_epochs_per_val,
            loader,
            loss_fncs,
            model,
            model_kwargs,
            start=step,
            label=label,
            **kwargs,
        )

        utils.save_checkpoint(ckpt, step, model=model)
        val_dict, val_out_dir = infer_model(
            step, val_loader, model, model_kwargs, loss_fncs, label=label
        )
        save_grid_vis(val_out_dir, val_dict)

    return step, val_dict


def update_config(cfg, loader):
    """
    we provide a min number of iterations for each phase,
    need to update the config to reflect this
    """
    N = len(loader) * cfg.batch_size
    for phase, epochs in cfg.epochs_per_phase.items():
        n_iters = cfg.iters_per_phase[phase]
        cfg.epochs_per_phase[phase] = max(n_iters // N + 1, epochs)

    if cfg.n_layers <= 2:
        cfg.w_kmeans *= 0.1
        cfg.epochs_per_phase["kmeans"] = cfg.epochs_per_phase["kmeans"] // 10

    # also update the vis and val frequency in iterations
    cfg.vis_every = max(cfg.vis_every, cfg.vis_epochs * N)
    cfg.val_every = max(cfg.val_every, cfg.val_epochs * N)
    print("epochs_per_phase", cfg.epochs_per_phase)
    print("vis_every", cfg.vis_every)
    print("val_every", cfg.val_every)
    return cfg


def save_metric(out_dir, out_dict, name="ious"):
    os.makedirs(out_dir, exist_ok=True)
    if name not in out_dict:
        return

    vec = out_dict[name].detach().cpu()
    if len(vec.shape) > 2:
        return

    ok = (vec >= 0).all(dim=-1)
    vec = vec[ok]
    np.savetxt(os.path.join(out_dir, f"frame_{name}.txt"), vec)
    np.savetxt(os.path.join(out_dir, f"mean_{name}.txt"), vec.mean(dim=0))
    print(name, vec.mean(dim=0))


def compute_multiple_iou(batch_in, batch_out):
    """
    :param masks (B, M, *, H, W)
    :param gt (B, C, H, W)
    :returns iou (B, M) chooses the best iou for each mask
    """
    if "gt" not in batch_in:
        return batch_out

    gt, ok = batch_in["gt"]
    if ok.sum() < 1:
        return batch_out

    with torch.no_grad():
        masks = batch_out["masks"]

        B, C, H, W = gt.shape
        masks_bin = masks.view(B, -1, 1, H, W) > 0.5
        gt_bin = gt.view(B, 1, C, H, W) > 0.5
        ious = utils.compute_iou(masks_bin, gt_bin, dim=(-1, -2))  # (B, M, C)
        ious = ious.amax(dim=-1)  # (B, M)
        ious[~ok] = -1

    batch_out["ious"] = ious
    return batch_out


def save_grid_vis(out_dir, vis_dict, pad=4, save_dirs=False):
    save_keys = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in save_keys):
        print(f"not all keys in vis_dict, cannot save to {out_dir}")
        return

    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}
    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid_vis(vis_dict, pad=pad)
    grid_path = os.path.join(out_dir, "grid_vis.mp4")
    utils.save_vid(grid_path, grid)

    if save_dirs:
        for save in save_keys:
            save_dir = os.path.join(out_dir, save)
            utils.save_batch_img_dir(save_dir, vis_dict[save])


def make_grid_vis(vis_dict, pad=4):
    """
    make panel vis with input, layers, view_vis, textures, and recon
    :param rgb (B, 3, H, W)
    :param recons (B, 3, H, W)
    :param layers (B, M, 3, H, W)
    :param texs (1, M, 3, H, W)
    :param view_vis (B, M, 3, H, W)
    """
    required = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in required):
        print(f"not all keys in vis_dict, cannot make grid vis")
        return

    rgb = vis_dict["rgb"]
    N, _, h, w = rgb.shape
    texs_rs = TF.resize(
        vis_dict["texs"][0], size=(h, w), antialias=True
    )  # (M, 3, h, w)
    texs_rs = texs_rs[None].repeat(N, 1, 1, 1, 1)  # (N, M, 3, h, w)

    texs_vert = pad_cat_groups_vert(texs_rs, pad=pad)
    layers_vert = pad_cat_groups_vert(vis_dict["layers"], pad=pad)
    tforms_vert = pad_cat_groups_vert(vis_dict["view_vis"], pad=pad)

    N, _, H, _ = texs_vert.shape
    diff = (H - h) // 2
    rgb_pad = TF.pad(rgb, (0, diff, pad, H - h - diff), fill=1)
    recon_pad = TF.pad(vis_dict["recons"], (pad, diff, 0, H - h - diff), fill=1)

    final = torch.cat([rgb_pad, texs_vert, tforms_vert, layers_vert, recon_pad], dim=-1)
    return final


def pad_cat_groups_vert(tensor, pad=4):
    """
    :param tensor (B, M, 3, h, w)
    :param pad (int)
    """
    padded = TF.pad(tensor, pad, fill=1)  # (B, M, 3, h+2*pad, w+2*pad)
    B, M, C, H, W = padded.shape
    catted = padded.transpose(1, 2).reshape(B, C, -1, W)
    return catted[..., pad:-pad, :]  # remove top-most and bottom-most padding
