import os
import glob
import numpy as np
from PIL import Image
from concurrent import futures


"""
Reformats DAVIS annotations to write separate masks for each object
"""


def reformat_masks(mask_root, out_root, seq, res="480p"):
    mask_dir = os.path.join(mask_root, res, seq)
    out_dir = os.path.join(out_root, res, seq)
    os.makedirs(out_dir, exist_ok=True)
    print("writing masks in {} to {}".format(mask_dir, out_dir))

    mask_files = sorted(glob.glob("{}/*.png".format(mask_dir)))
    og_imgs = [np.array(Image.open(f)) for f in mask_files]
    n_objs = max([img.max() for img in og_imgs])
    for i in range(1, n_objs + 1):
        subdir = os.path.join(out_dir, str(i))
        os.makedirs(subdir, exist_ok=True)
        for src_path, img in zip(mask_files, og_imgs):
            name = os.path.basename(src_path)
            out_path = os.path.join(subdir, name)
            mask = Image.fromarray(255 * (img == i).astype(np.uint8))
            mask.save(out_path)
    print(
        "wrote {} img masks for {} objects for seq {}".format(
            len(mask_files), n_objs, seq
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", default="/home/vye/data/DAVIS/Annotations_multi")
    parser.add_argument("--out_root", default="/home/vye/data/DAVIS/Annotations_groups")
    parser.add_argument("--res", default="480p")
    parser.add_argument("-j", "--n_workers", type=int, default=8)
    args = parser.parse_args()

    mask_dir = os.path.join(args.mask_root, args.res)
    seqs = sorted(os.listdir(mask_dir))

    with futures.ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for seq in seqs:
            ex.submit(reformat_masks, args.mask_root, args.out_root, seq, args.res)
