import os
import glob
import numpy as np
from PIL import Image

import subprocess
from concurrent import futures

BASE_DIR = os.path.abspath("__file__/..")
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys

sys.path.append(os.path.join(ROOT_DIR, "src"))
from data import read_flo, write_flo


"""
script that computes backward flow for sintel in hi-res,
then resizes everything to 480p
"""


def process_sequence(gpu, src_root, tgt_root, seq, gap=-1, width=480, height=-1):
    img_dirs_resize = ["final", "clean", "occlusions"]
    for img_dir in img_dirs_resize:
        ok = batch_resize_images(
            src_root, tgt_root, img_dir, seq, width=width, height=height
        )
        if not ok:
            print(f"resize {img_dir} failed for seq {seq}")

    ok = batch_resize_flows(src_root, tgt_root, "flow", seq, width, height)
    if not ok:
        print("RESIZE FLOW FAILED FOR SEQ", seq)

    ok, bck_dir = compute_backward_flow(gpu, src_root, seq, gap)
    if not ok:
        print("COMPUTE BACK FLOW FAILED FOR SEQ", seq)
        return

    ok = batch_resize_flows(src_root, tgt_root, "back_flow", seq, width, height)
    if not ok:
        print("RESIZE BACK FLOW FAILED FOR SEQ", seq)


def compute_backward_flow(gpu, root, seq, gap=-1):
    gap = -abs(gap)
    exe = os.path.join(BASE_DIR, "run_raft.py")
    rgb = os.path.join(root, "clean", seq)
    out = os.path.join(root, "back_flow", seq)
    out_img = os.path.join(root, "back_flow_viz", seq)
    cmd = f"python {exe} {rgb} {out} -I {out_img} --gap {gap} -b 1"
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True)
    ok = check_flows(rgb, out, gap)
    return ok, out


def check_flows(rgb_dir, flow_dir, gap):
    if not os.path.isdir(rgb_dir):
        print(f"{rgb_dir} does not exist")
        return False
    if not os.path.isdir(flow_dir):
        print(f"{flow_dir} does not exist")
        return False
    n_rgb = len(os.listdir(rgb_dir))
    n_flo = len(os.listdir(flow_dir))
    return n_rgb == n_flo + abs(gap)


def batch_resize_images(src_root, tgt_root, subd, seq, width=480, height=-1):
    assert width > 0 or height > 0, (width, height)
    src_dir = os.path.join(src_root, subd, seq)
    if not os.path.isdir(src_dir):
        print(f"{src_dir} does not exist")
        return False

    src_paths = f"{src_dir}/frame_%04d.png"

    tgt_dir = os.path.join(tgt_root, subd, seq)
    tgt_paths = f"{tgt_dir}/frame_%04d.png"
    os.makedirs(tgt_dir, exist_ok=True)

    scale_str = ""
    if width > 0:
        scale_str = f"scale={width}:-1"
    else:
        scale_str = f"scale=-1:height"

    cmd = f"ffmpeg -f image2 -i {src_paths} -vf {scale_str} {tgt_paths} -loglevel error -n"
    print(cmd)
    subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)

    n_src = len(os.listdir(src_dir))
    n_tgt = len(os.listdir(tgt_dir))
    return n_src == n_tgt


def batch_resize_flows(src_root, tgt_root, subd, seq, width=480, height=-1):
    assert width > 0 or height > 0
    src_dir = os.path.join(src_root, subd, seq)
    if not os.path.isdir(src_dir):
        print(f"{src_dir} does not exist")
        return False

    tgt_dir = os.path.join(tgt_root, subd, seq)
    os.makedirs(tgt_dir, exist_ok=True)

    src_names = sorted(os.listdir(src_dir))

    for name in src_names:
        src_path = os.path.join(src_dir, name)
        tgt_path = os.path.join(tgt_dir, name)
        resize_flow(src_path, tgt_path, width=width, height=height)

    n_src = len(os.listdir(src_dir))
    n_tgt = len(os.listdir(tgt_dir))
    return n_src == n_tgt


def resize_flow(src_path, tgt_path, width=480, height=-1):
    assert width > 0 or height > 0
    flow = read_flo(src_path)
    H, W, _ = flow.shape

    if width > 0:
        scale = width / W
        height = int(scale * H)
    else:
        scale = height / H
        width = int(scale * W)
    u = Image.fromarray(flow[..., 0]).resize((width, height), Image.ANTIALIAS)
    v = Image.fromarray(flow[..., 1]).resize((width, height), Image.ANTIALIAS)
    u, v = scale * np.array(u), scale * np.array(v)
    flow_rs = np.stack([u, v], axis=-1)
    write_flo(tgt_path, flow_rs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=[0])
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--gap", type=int, default=-1)
    args = parser.parse_args()

    src_root = "/home/vye/data/sintel-full/training"
    tgt_root = "/home/vye/data/sintel-full/training_480p"
    seqs = sorted(os.listdir(f"{src_root}/final"))
    #     with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
    #         for i, seq in enumerate(seqs):
    #             gpu = i % len(args.gpus)
    #             exe.submit(process_sequence, gpu, src_root, tgt_root, seq, gap=args.gap, width=args.width, height=args.height)

    for i, seq in enumerate(seqs):
        gpu = i % len(args.gpus)
        process_sequence(
            gpu,
            src_root,
            tgt_root,
            seq,
            gap=args.gap,
            width=args.width,
            height=args.height,
        )
