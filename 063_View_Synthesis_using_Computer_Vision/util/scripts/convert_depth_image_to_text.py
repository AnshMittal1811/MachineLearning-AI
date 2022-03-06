import os
import numpy as np
from PIL import Image

from tqdm.auto import tqdm

def load_depth(file):
    image = Image.open(file).convert("L") # convert to grayscale
    pixels = image.load() # .load() returns an object with access to pixel values as [x, y]
    shape = image.size

    return image, pixels, shape


def main(input, output):
    depth_files = sorted([os.path.join(input, f) for f in os.listdir(input) if f.endswith('.depth.png')])

    print("Converting {} depth files".format(len(depth_files)))

    for file in tqdm(depth_files):
        image, pixels, shape = load_depth(file)
        out_path = os.path.join(output, file[:-4]) # remove .png and save as .depth file

        #import matplotlib.pyplot as plt
        #plt.imshow(image)
        #plt.show()

        out_file = open(out_path, "w")
        row, column = shape
        for y in range(column):
            for x in range(row):
                pixel = pixels[x, y] / 255.0
                out_file.write(str(pixel) + ' ')
            out_file.write("\n")
        out_file.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert .depth text files float files to binary numpy .npy files')
    parser.add_argument('--input', metavar='path', required=True,
                        help='path/to/input/directory')
    parser.add_argument('--output', metavar='path', required=False, default=None,
                        help='path/to/output/directory. Optional, default: input directory')

    args = parser.parse_args()
    main(input=args.input,
         output=args.output if args.output is not None else args.input)
