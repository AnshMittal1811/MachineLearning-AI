import os
import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s", help="source folder", type=str, required=True)
parser.add_argument("--t", help="target folder", type=str, required=True)
parser.add_argument("--h", help="target height", type=int, required=True)
parser.add_argument("--w", help="target width", type=int, required=True)
args = parser.parse_args()

source_folder = args.s
target_folder = args.t
target_res = (args.w, args.h) # W x H

scenes = dir_list = os.listdir(source_folder)

for scene in scenes:
    if not os.path.exists(os.path.join(target_folder, scene)):
        os.makedirs(os.path.join(target_folder, scene))
    if not os.path.exists(os.path.join(target_folder, scene, "images")):
        os.makedirs(os.path.join(target_folder, scene, "images"))
    
    img_names = []
    for img_name in os.listdir(os.path.join(source_folder, scene, "images")):
        img_path = os.path.join(source_folder, scene, "images", img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_res)
        cv2.imwrite(os.path.join(target_folder, scene, "images", img_name), img)
        img_names.append(img_name + '\n')

    with open(os.path.join(target_folder, scene, 'train.txt'), 'w') as f:
        f.writelines(img_names)