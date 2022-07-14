import os
import numpy as np
import uuid
import shutil
from pathlib import Path

from tqdm.auto import tqdm

def find_pairs(input):
    print("Parse all available images...")
    pairs = []
    for scan in tqdm(os.listdir(input)):
        scan_path = os.path.join(input, scan)
        if os.path.isdir(scan_path):
            for f in os.listdir(scan_path):
                if f.endswith("_0.png"):
                    rgb_name = f
                    rgb_path = os.path.join(scan_path, rgb_name)

                    seg_name = f[:-3] + "seg.png"
                    seg_path = os.path.join(scan_path, seg_name)

                    pairs.append((seg_name, seg_path, rgb_name, rgb_path))
    print(f"Found {len(pairs)} seg2rgb pairs")

    return pairs

def copy_selected_pairs(output, selected_pairs):
    # generate random id for this sample output
    id = uuid.uuid4()

    # output folder is output/id
    out_dir = os.path.join(output, str(id))

    # folder for seg as specified in: https://github.com/NVIDIA/pix2pixHD
    seg_dir = os.path.join(out_dir, "train_A")
    Path(seg_dir).mkdir(parents=True, exist_ok=True)

    # folder for rgb as specified in: https://github.com/NVIDIA/pix2pixHD
    rgb_dir = os.path.join(out_dir, "train_B")
    Path(rgb_dir).mkdir(parents=True, exist_ok=True)

    # copy seg + rgb
    print(f"Copy {len(selected_pairs)} seg2rgb pairs to {out_dir}...")
    for seg_name, seg_path, rgb_name, rgb_path in tqdm(selected_pairs):
        shutil.copy(seg_path, seg_dir)
        shutil.copy(rgb_path, rgb_dir)
    print(f"Copied {len(selected_pairs)} seg2rgb pairs successfully")

def main(input, output, samples, seed):

    # find all possible seg2rgb pairs
    pairs = find_pairs(input)

    # shuffle list
    np.random.seed(seed)
    np.random.shuffle(pairs)

    # select <samples> many pairs
    selected_pairs = pairs[:samples]

    # copy them to output dir
    copy_selected_pairs(output, selected_pairs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Select n seg2rgb pairs from mp3d dataset for pix2pix training')
    parser.add_argument('--input', metavar='path', required=True,
                        help='path/to/mp3d')
    parser.add_argument('--output', metavar='path', required=True,
                        help='path/to/output/directory. Where to store the selected images')
    parser.add_argument('--samples', metavar='N', type=int, required=False, default=200000,
                        help='how many random pairs to select')
    parser.add_argument('--seed', metavar='N', type=int, required=False, default=42,
                        help='random seed for selecting n random samples from all available samples')

    args = parser.parse_args()
    main(input=args.input,
         output=args.output,
         samples=args.samples,
         seed=args.seed)
