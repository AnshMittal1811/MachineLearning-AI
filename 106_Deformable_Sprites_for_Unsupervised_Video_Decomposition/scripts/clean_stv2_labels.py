import os
import glob
import shutil


"""
Reformats segtrack annotations to always have subdirectories, even with single object
"""


def main(root_dir):
    seq_dirs = glob.glob("{}/**/".format(root_dir))
    for seq_dir in seq_dirs:
        reformat_gt(seq_dir)


def reformat_gt(gt_dir):
    # check if images are here
    print("reformatting", gt_dir)
    impaths = glob.glob("{}/*.png".format(gt_dir))
    if len(impaths) < 1:
        # already organized into subdirectories
        return
    subdir = os.path.join(gt_dir, "1")
    os.makedirs(subdir, exist_ok=True)
    for path in impaths:
        name = os.path.basename(path)
        target = os.path.join(subdir, name)
        shutil.move(path, target)
    print("moved {} image files to {}".format(len(impaths), subdir))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--src_dir", default="/home/vye/data/SegTrackv2/GroundTruth"
    )
    args = parser.parse_args()

    main(args.src_dir)
