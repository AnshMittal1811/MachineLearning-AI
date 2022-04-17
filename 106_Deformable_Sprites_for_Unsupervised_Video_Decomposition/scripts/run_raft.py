import argparse
import imageio
import glob
import numpy as np
import os
from PIL import Image

from concurrent import futures

import torch

RAFT_BASE = os.path.expanduser("~/RAFT")

import sys

sys.path.append(os.path.join(RAFT_BASE, "core"))

from raft import RAFT
from utils.utils import InputPadder
from utils.flow_viz import flow_to_image

sys.path.append(os.path.abspath("__file__/../../src"))
from data import write_flo


DEVICE = torch.device("cuda")


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def is_image(path):
    path = path.lower()
    ext = os.path.splitext(path)[-1]
    return ext == ".png" or ext == ".jpg" or ext == ".bmp"


def run_raft(args):
    if args.out_img_dir is None:
        args.out_img_dir = os.path.join(args.out_dir, "vis")

    imfiles = sorted(filter(is_image, glob.glob(f"{args.rgb_dir}/*")))
    if len(imfiles) < 1:
        print("NO IMAGES FOUND IN", args.rgb_dir)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_img_dir, exist_ok=True)
    n_raw = len(os.listdir(args.out_dir))
    if n_raw == len(imfiles) - abs(args.gap):
        print("already {} flows existing in {}".format(n_raw, args.out_dir))
        return

    print("Running RAFT on", args.rgb_dir)
    print("Writing flows to", args.out_dir, args.out_img_dir)

    model = torch.nn.DataParallel(RAFT(args))
    model_path = os.path.join(RAFT_BASE, "models", args.ckpt)
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to(DEVICE)
    model.eval()

    images = torch.cat([load_image(f) for f in imfiles], dim=0)

    padder = InputPadder(images.shape)
    print("prepad image shapes", images.shape)
    images = padder.pad(images)[0]
    print("postpad image shapes", images.shape)

    if args.gap > 0:
        src_idcs = torch.arange(len(images) - args.gap)
    elif args.gap < 0:
        src_idcs = torch.arange(-args.gap, len(images))
    else:
        raise ValueError(
            "Must provide nonzero gap for flow computation, gave", args.gap
        )
    tgt_idcs = src_idcs + args.gap

    src_imgs, tgt_imgs = images[src_idcs], images[tgt_idcs]
    names = [os.path.splitext(os.path.basename(imfiles[i]))[0] for i in src_idcs]

    flow_batches = []
    with torch.no_grad():
        for i in range(0, len(src_imgs), args.batch_size):
            src_batch = src_imgs[i : i + args.batch_size].to(DEVICE)
            tgt_batch = tgt_imgs[i : i + args.batch_size].to(DEVICE)

            flows_low, flows_up = model(src_batch, tgt_batch, iters=20, test_mode=True)

            flows = padder.unpad(flows_up)
            flows = flows.detach().cpu()  # (N, 2, H, W)
            flow_batches.append(flows)

    flows = torch.cat(flow_batches, axis=0).permute(0, 2, 3, 1).numpy()  # (N, H, W, 2)
    print("processed {} flows".format(len(flows)))
    print(flows.shape)

    with futures.ProcessPoolExecutor(max_workers=args.n_writers) as exe:
        for i, name in enumerate(names):
            raw_path = os.path.join(args.out_dir, "{}.flo".format(name))
            exe.submit(write_flo, raw_path, flows[i])

            flow_img = flow_to_image(flows[i]).astype(np.uint8)
            img_path = os.path.join(args.out_img_dir, "{}.png".format(name))
            exe.submit(imageio.imwrite, img_path, flow_img)

    print("Wrote {} raw flows to {}".format(len(flows), args.out_dir))
    print("Wrote {} flow images to {}".format(len(flows), args.out_img_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_dir")
    parser.add_argument("out_dir")
    parser.add_argument("-I", "--out_img_dir", default=None)
    parser.add_argument("--ckpt", default="raft-things.pth", help="restore checkpoint")
    parser.add_argument("--gap", type=int, default=1, help="default 1")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="default 8")
    parser.add_argument("-j", "--n_writers", type=int, default=8, help="default 8")

    ## include RAFT model args
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    run_raft(args)
