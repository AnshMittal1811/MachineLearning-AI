from PIL import Image
import sys
import numpy as np
import os
import math
from scipy.misc import imsave
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=None, help='The input folder path')
parser.add_argument('--b', default=0.2, type=float, help='The input baseline')
args = parser.parse_args()

# path 
path = args.path
if path[-1]=='/':
    path = path[:-1]
out_path = '%s_depth'%path
if not os.path.isdir(out_path):
    os.makedirs(out_path)   # build the dir for the disparity output

disp_lst = [ x for x in sorted( os.listdir(path))] # the list for the depth name

b = args.b  # camera baseline

# angle array for calculation
angle = np.zeros((512,1024))
angle2 = np.zeros((512, 1024))
for i in range(1024):
    for j in range(512):
        theta_T = math.pi - ((j + 0.5) * math.pi/ 512)
        angle[j,i] = b* math.sin(theta_T)
        angle2[j,i] = b * math.cos(theta_T)

for i in tqdm(range(len(disp_lst)), desc="image"):
    disp = np.load('%s/%s'%(path, disp_lst[i]))# /512 * 180
    w = disp.shape[1]
    h = disp.shape[0]
    mask0 = disp==0
    maskn0 = disp>0

    de_pred = np.zeros((h,w))
    de_pred[maskn0] = (angle[maskn0] / np.tan(disp[maskn0]/180*math.pi)) + angle2[maskn0]
    de_pred[mask0] = 0
    de_name = ""
    de_name = disp_lst[i]
    np.save("%s/%s"%(out_path, de_name),de_pred)

