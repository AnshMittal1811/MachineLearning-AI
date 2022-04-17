import os
import glob
import imageio

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import sys

sys.path.append(os.path.abspath("__file__/../../src"))
import utils


def propagate_edits(src_dir, out_name, edit_idcs=None, ext="mp4", pad=8):
    out_dir = os.path.join(src_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    rgbs, masks, coords, texs = load_components(src_dir)
    edited, idcs = load_edit_textures(src_dir, texs, edit_idcs)  # (M, 3, H, W)

    M, C, H, W = texs.shape
    rgb_rs = TF.resize(rgbs, (H, W), antialias=True)

    ## make grid col [tex_old] col [tex_edit] col [composite]
    tex_old_pad = [TF.pad(texs[i], pad, fill=1) for i in idcs]
    tex_old_cat = torch.cat(tex_old_pad, dim=-2)[:, pad:-pad, pad:]

    tex_edit_pad = [TF.pad(edited[i], pad, fill=1) for i in idcs]
    tex_edit_cat = torch.cat(tex_edit_pad, dim=-2)[:, pad:-pad]

    tex_cat = torch.cat([tex_old_cat, tex_edit_cat], dim=-1)

    out_path = f"{out_dir}/edit_comp.{ext}"
    grid_path = f"{out_dir}/grid_vis.{ext}"
    layer_paths = [f"{out_dir}/edit_layers_{i}.{ext}" for i in range(M)]

    comp_writer = imageio.get_writer(out_path, format=ext)
    layer_writers = [imageio.get_writer(p, format=ext) for p in layer_paths]
    grid_writer = imageio.get_writer(grid_path, format=ext)

    for i in range(len(coords)):
        appr = F.grid_sample(edited, coords[i], align_corners=False)  # (M, 3, h, w)
        layers = masks[i] * appr  # (M, 3, h, w)
        comp = layers.sum(dim=0)  # (3, h, w)
        comp_writer.append_data((255 * comp.permute(1, 2, 0)).byte().numpy())

        comp_rs = TF.resize(comp, (H, W), antialias=True)
        comp_pad = TF.pad(comp_rs, [pad, 0, 0, 0], fill=1)
        rgb_pad = TF.pad(rgb_rs[i], [0, 0, pad, 0], fill=1)
        grid = torch.cat([rgb_pad, tex_cat, comp_pad], dim=2)
        grid_writer.append_data((255 * grid.permute(1, 2, 0)).byte().numpy())

        layer_vis = utils.composite_rgba_checkers(masks[i], layers)
        for j, writer in enumerate(layer_writers):
            writer.append_data((255 * layer_vis[j].permute(1, 2, 0)).byte().numpy())


def load_coords_precomputed(src_dir, precomputed=True):
    """
    loads the masks, textures, coords from src dir
    """
    if precomputed:
        coord_path = os.path.join(src_dir, "coords.pth")
        coords = torch.load(coord_path).float()
        return coords

    ## TODO load from saved model checkpoint
    raise NotImplementedError


def load_masks(src_dir):
    mask_paths = sorted(glob.glob(f"{src_dir}/masks_[0-9]*.gif"))
    idcs = torch.tensor([get_index(p) for p in mask_paths])
    masks = torch.from_numpy(
        np.stack([np.stack(imageio.mimread(p), axis=0) for p in mask_paths], axis=1)
    ).float()  # (n, m, h, w)
    masks = masks[:, idcs].unsqueeze(2) / 255  # (n, m, 1, h, w)
    return masks


def load_rgbs(src_dir):
    rgb_paths = sorted(glob.glob(f"{src_dir}/recons_[0-9]*.gif"))
    if len(rgb_paths) != 1:
        print("found {} matching rgbs, need {}".format(len(rgb_paths), 1))
        raise ValueError
    rgbs = torch.from_numpy(np.stack(imageio.mimread(rgb_paths[0]), axis=0))
    rgbs = rgbs.float().permute(0, 3, 1, 2)[:, :3] / 255  # (n, 3, h, w)
    return rgbs


def load_textures(src_dir):
    tex_paths = sorted(glob.glob(f"{src_dir}/texs_[0-9]*.png"))
    idcs = torch.tensor([get_index(p) for p in tex_paths])
    texs = torch.stack(
        [torch.from_numpy(imageio.imread(p) / 255) for p in tex_paths], dim=0
    ).float()  # (m, h, w, 3)
    texs = texs[idcs].permute(0, 3, 1, 2)  # (m, 3, h, w)
    print("texs shape", texs.shape)
    return texs


def load_edit_textures(src_dir, texs, idcs=None):
    M, _, H, W = texs.shape
    tex_paths = sorted(glob.glob(f"{src_dir}/edit*_[0-9]*.png"))
    # parse names for which textures to replace
    all_idcs = [get_index(p) for p in tex_paths]
    idcs = all_idcs
    assert all(i < M for i in idcs)

    edits = torch.clone(texs)
    for i, edit_i in enumerate(idcs):
        path = tex_paths[i]
        print(f"loaded {edit_i}-th texture from {path}")
        tex = torch.from_numpy(imageio.imread(path) / 255).permute(2, 0, 1).float()[:3]
        tex = pad_diff(tex, edits[edit_i])
        edits[i] = tex

    return edits, idcs


def pad_diff(src, tgt):
    H, W = tgt.shape[-2:]
    h, w = src.shape[-2:]
    pl = (W - w) // 2
    pt = (H - h) // 2
    pr = W - w - pl
    pb = H - h - pt
    return TF.pad(src, (pl, pt, pr, pb), fill=1)


def resize_batch(tensor, scale):
    if scale != 1:
        *dims, c, h, w = tensor.shape
        H, W = scale * h, scale * w
        tensor_rs = tensor.view(-1, c, h, w)
        tensor_rs = TF.resize(tensor_rs, (H, W), antialias=True)
        return tensor_rs.view(*dims, c, H, W)
    return tensor


def get_index(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("--out_name", default="edited")
    parser.add_argument("--edit_idcs", default=None, nargs="*", type=int)
    args = parser.parse_args()

    propagate_edits(args.src_dir, args.out_name, args.edit_idcs)
