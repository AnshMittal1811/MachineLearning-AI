import os
import glob
import numpy as np
import imageio


"""
Reformats FBMS annotations from ppm/pgm files and writes separate masks for each object
"""


def main(root_dir, out_name):
    seq_dirs = glob.glob("{}/**/".format(root_dir))
    for seq_dir in seq_dirs:
        reformat_gt(seq_dir, out_name)


def count_dir(path):
    if not os.path.isdir(path):
        return 0
    return len(os.listdir(path))


def reformat_gt(data_dir, out_name):
    print("reformatting ground truth for", data_dir)
    gt_dir = os.path.join(data_dir, "GroundTruth")
    out_dir = os.path.join(data_dir, out_name)
    # first try looking at ppm extensions
    ok = reformat_ppm_seq(gt_dir, out_dir)
    if not ok:
        print("no ppm files found, trying pgm")
        # then look at pgm extensions if necessary
        ok = reformat_pgm_seq(gt_dir, out_dir)
    if not ok:
        print("COULD NOT GENERATE CLEAN LABELS FOR", data_dir)
    else:
        print("successfully cleaned labels for", data_dir)


def reformat_ppm_seq(gt_dir, out_dir):
    """ """
    ppms = glob.glob("{}/*_gt.ppm".format(gt_dir))
    ppms = list(filter(lambda x: "PROB" not in x, ppms))
    if len(ppms) < 1:
        return False

    recoded = []
    frame_names = []
    for gt_path in ppms:
        name = os.path.basename(gt_path).split("_gt.ppm")[0]
        gt = imageio.imread(gt_path) / 255.0
        code = gt[..., 0] * 4 + gt[..., 1] * 2 + gt[..., 0]
        recoded.append(code)
        frame_names.append(name)

    recoded = np.stack(recoded, axis=0)  # (N, H, W)
    vals = np.unique(recoded)

    # background is white, skip the last value
    for i, v in enumerate(vals[:-1]):
        subdir = os.path.join(out_dir, str(i + 1))
        os.makedirs(subdir, exist_ok=True)
        for j, name in enumerate(frame_names):
            mask = (recoded[j] == v).astype(np.uint8)
            path = os.path.join(subdir, "{}.png".format(name))
            imageio.imwrite(path, (255 * mask))
        print("wrote {} frames to {}".format(len(recoded), subdir))
    return True


def reformat_pgm_seq(gt_dir, out_dir):
    pgms = glob.glob("{}/*.pgm".format(gt_dir))
    if len(pgms) < 1:
        return False

    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in pgms]
    gts = np.stack([imageio.imread(p) for p in pgms], axis=0)  # (N, H, W)
    vals = np.unique(gts)

    # background is black, skip the first value
    for i, v in enumerate(vals[1:]):
        subdir = os.path.join(out_dir, str(i))
        os.makedirs(subdir, exist_ok=True)
        for j, name in enumerate(frame_names):
            mask = (gts[j] == v).astype(np.uint8)
            path = os.path.join(subdir, "{}.png".format(name))
            imageio.imwrite(path, (255 * mask))
        print("wrote {} frames to {}".format(len(gts), subdir))
    return True


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--src_dir", default="/home/vye/data/FBMS_Testset")
    parser.add_argument("-o", "--out_name", default="GroundTruthClean")
    args = parser.parse_args()

    main(args.src_dir, args.out_name)
