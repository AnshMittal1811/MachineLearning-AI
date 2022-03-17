import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, default='../../000_src_data/vid215_201712271749')
parser.add_argument('-o', '--output', type=str, default='01_cut/cut_vid215_201801031127')
parser.add_argument('-f', '--fps', type=int, default=30)
parser.add_argument('--ss', type=int, default=1)
parser.add_argument('--t', type=int, default=8, help='negative value, means how long to the eof ')

args = parser.parse_args()

# in_path = '../../000_src_data/data_vid10_1218'

# out_path = '../../011_aug_data/cutdata_vid10_1218'

in_path = args.input
out_path = args.output
fps = args.fps

if not os.path.exists(out_path):
    os.makedirs(out_path)

files = os.listdir(in_path)
files.sort()

for file in files:
    os.mkdir('{}/{}'.format(out_path, file))
    os.system(
        'ffmpeg -ss {} -t {} -r {} -i {}/{} {}/{}/%4d.png'.format(args.ss, args.t, fps, in_path, file, out_path, file)
    )

