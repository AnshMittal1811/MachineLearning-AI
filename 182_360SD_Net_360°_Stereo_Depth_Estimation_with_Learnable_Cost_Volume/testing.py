from __future__ import print_function
import os
import time

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import skimage
import skimage.io
import skimage.transform
import numpy as np
from tqdm import tqdm

from dataloader import preprocess
from models import LCV_ours_sub3
from dataloader import testing_loader as DA

parser = argparse.ArgumentParser(description='360SD-Net Testing')
parser.add_argument('--datapath',
                    default='data/MP3D/test/',
                    help='select model')
parser.add_argument('--checkpoint',
                    default=None,
                    help='load checkpoint path')
parser.add_argument('--model',
                    default='360SDNet',
                    help='select model')
parser.add_argument('--maxdisp',
                    type=int,
                    default=68,
                    help='maxium disparity')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--real',
                    action='store_true',
                    default=False,
                    help='adapt to real world images as input')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outfile',
                    type=str,
                    help='the output path to put the output disparity')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Random Seed ------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# ----------------------------

# Load data -----------------
test_up_img, test_down_img = DA.dataloader(args.datapath)
# ----------------------------

# Load model ----------------
if args.model == '360SDNet':
    print("Load model 360SD-Net")
    model = LCV_ours_sub3(args.maxdisp)
else:
    raise NotImplementedError('Model Not Implemented!!!')
# ----------------------------

# Model Multi-GPU -----------
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
# ----------------------------

# Real World inference ------
if args.real:
    print("Real World Testing!!!")
# ----------------------------

# Load checkpoint --------------------------------
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
# -------------------------------------------------

# Create Angle Info -----------------------------------------------
angle_y = np.array([(i - 0.5) / 512 * 180 for i in range(256, -256, -1)])
angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 1024, 1))
equi_info = angle_ys
# ------------------------------------------------------------------


# Testing Fuction ---------------------------------------------------------
def test(imgU, imgD):
    model.eval()
    # cuda?
    if args.cuda:
        imgU = torch.FloatTensor(imgU).cuda()
        imgD = torch.FloatTensor(imgD).cuda()
    imgU, imgD = Variable(imgU), Variable(imgD)

    with torch.no_grad():
        output = model(imgU, imgD)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp
# --------------------------------------------------------------------------


# Main Fuction ------------------------------------------------------------
def main():
    processed = preprocess.get_transform(augment=False)
    print("Start Testing!!!")
    # print(args.pad)
    total_time = 0

    for inx in tqdm(range(len(test_up_img))):
        # read grey scale
        if args.real:
            imgU_o = np.tile(
                skimage.io.imread(test_up_img[inx], as_grey=True)[:, :,
                                                                  np.newaxis],
                (1, 1, 3)) * 255
            imgD_o = np.tile(
                skimage.io.imread(test_down_img[inx],
                                  as_grey=True)[:, :, np.newaxis],
                (1, 1, 3)) * 255
        else:
            imgU_o = (skimage.io.imread(test_up_img[inx]))
            imgD_o = (skimage.io.imread(test_down_img[inx]))

        # concatenate polar angle as equirectangular information --------
        imgU_o = np.concatenate([imgU_o, equi_info], 2)
        imgD_o = np.concatenate([imgD_o, equi_info], 2)

        # Real World / Synthetic preprocessing --------------------------
        if args.real:
            compose_trans = transforms.Compose([transforms.ToTensor()])
            imgU = compose_trans(imgU_o).numpy()
            imgD = compose_trans(imgD_o).numpy()
            imgU = np.reshape(imgU, [1, 4, imgU.shape[1], imgU.shape[2]])
            imgD = np.reshape(imgD, [1, 4, imgD.shape[1], imgD.shape[2]])
        else:
            imgU = processed(imgU_o).numpy()
            imgD = processed(imgD_o).numpy()
            imgU = np.reshape(imgU, [1, 4, imgU.shape[1], imgU.shape[2]])
            imgD = np.reshape(imgD, [1, 4, imgD.shape[1], imgD.shape[2]])

        # Wide padding -----------------------------
        LR_pad = 32
        imgU = np.lib.pad(imgU, ((0, 0), (0, 0), (0, 0), (LR_pad, LR_pad)),
                          mode='wrap')
        imgD = np.lib.pad(imgD, ((0, 0), (0, 0), (0, 0), (LR_pad, LR_pad)),
                          mode='wrap')

        # Testing and count time -------------------
        start_time = time.time()
        pred_disp = test(imgU, imgD)
        total_time += (time.time() - start_time)
        img = pred_disp[:, LR_pad:-LR_pad]

        # Save output ------------------------------
        if args.outfile[-1] == '/':
            args.outfile = args.outfile[:-1]
        os.system('mkdir -p %s' % args.outfile)
        np.save(
            args.outfile + '/' + test_up_img[inx].split('/')[-1][:-4] + '.npy',
            img)
        # -------------------------------------------

    # Print Total Time
    print("Total time: ", total_time, "Average time: ",
          total_time / len(test_up_img))
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    print("start main")
    main()
