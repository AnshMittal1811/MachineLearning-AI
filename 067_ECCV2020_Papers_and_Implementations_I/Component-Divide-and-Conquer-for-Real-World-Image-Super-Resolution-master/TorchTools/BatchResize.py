import os
import argparse

from PIL import Image

from DataTools.FileTools import _video_image_file, _image_file
from Functions.functional import resize

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, default='../../000_src_data/vid215_201712271749')
parser.add_argument('-o', '--output', type=str, default='01_cut/cut_vid215_201801031127')
parser.add_argument('-s', '--scala', type=int, default=2)
parser.add_argument('-v', type=bool, default=True)
parser.add_argument('--up', type=bool, default=False)
args = parser.parse_args()

input_path = os.path.abspath(args.input)
output_path = os.path.abspath(args.output)
if args.up:
    scala = 1 / args.scala
else:
    scala = args.scala
V = args.v


def make_dir(input, output):
    dirs = os.listdir(input)
    os.mkdir(output)
    for i in dirs:
        os.mkdir(os.path.join(output, i))


def resize_and_save(file_org, output, scala):
    im = Image.open(file_org)
    w, h = im.size
    im = resize(im, (int(h // scala), int(w // scala)))
    name = os.path.split(file_org)
    vdir = os.path.split(name[0])
    save_name = os.path.join(output, vdir[1])
    save_name = os.path.join(save_name, name[1])
    im.save(save_name)
    print(save_name)

if V:
    file_list = _video_image_file(input_path)
    make_dir(input_path, output_path)
    for j in file_list:
        for i in j:
            resize_and_save(i, output_path, scala)
else:
    file_list = _image_file(input_path)
    os.mkdir(output_path)
    for i in file_list:
        resize_and_save(i, output_path, scala)







