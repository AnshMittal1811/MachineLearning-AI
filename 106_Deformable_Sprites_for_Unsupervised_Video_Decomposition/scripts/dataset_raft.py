import os
import subprocess

from concurrent import futures

BASE_DIR = os.path.abspath("__file__/..")
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys

sys.path.append(os.path.join(ROOT_DIR, "src"))
from data import get_data_subdir, match_custom_seq


def process_sequence(gpu, dtype, root, seq, gap, res="480p", batch_size=4):
    if dtype == "fbms":
        rgb_name = ""
    elif dtype == "custom":
        rgb_name = "PNGImages"
        seq = match_custom_seq(root, rgb_name, seq)
    elif dtype == "davis" or dtype == "stv2":
        rgb_name = "JPEGImages"
    else:
        raise NotImplementedError

    print(rgb_name, seq)

    subds = [rgb_name, "raw_flows_gap{}".format(gap), "flow_imgs_gap{}".format(gap)]

    rgb, out, out_img = [get_data_subdir(dtype, root, sd, seq, res) for sd in subds]
    print(rgb, out, out_img)
    exe = os.path.join(BASE_DIR, "run_raft.py")
    cmd = f"python {exe} {rgb} {out} -I {out_img} --gap {gap} -b {batch_size}"
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True)


def main(args):
    if args.root is None:
        if args.dtype == "fbms":
            args.root = "/home/vye/data/FBMS_Testset"
        elif args.dtype == "davis":
            args.root = "/home/vye/data/DAVIS"
        elif args.dtype == "stv2":
            args.root = "/home/vye/data/SegTrackv2"
        elif args.dtype == "custom":
            args.root = "/home/vye/data/custom_videos"

    if args.seqs is None:
        if args.dtype == "fbms":
            args.seqs = os.listdir(args.root)
        elif args.dtype == "davis":
            args.seqs = os.listdir(os.path.join(args.root, "JPEGImages", args.dres))
        elif args.dtype == "stv2":
            args.seqs = os.listdir(os.path.join(args.root, "JPEGImages"))
        elif args.dtype == "custom":
            args.seqs = os.listdir(os.path.join(args.root, "PNGImages"))
        else:
            raise NotImplementedError

    i = 0
    with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
        for seq in args.seqs:
            for gap in [args.gap, -args.gap]:
                gpu = args.gpus[i % len(args.gpus)]
                ex.submit(
                    process_sequence,
                    gpu,
                    args.dtype,
                    args.root,
                    seq,
                    gap,
                    args.dres,
                    args.batch_size,
                )
                i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="path to dataset root folder")
    parser.add_argument(
        "--dtype", default="custom", choices=["custom", "davis", "fbms", "stv2"]
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="+", default=[0])
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--dres", default="480p")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    main(args)
