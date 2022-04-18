import sys
sys.path.append('../code')
import imageio
imageio.plugins.freeimage.download()

import torch
import torch.nn as nn
import numpy as np
import argparse
import imageio
import cv2
import os

from model.sg_render import compute_envmap

TINY_NUMBER = 1e-8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envmap_path', type=str, default='envmaps/envmap12.exr')
    parser.add_argument('--num_sg', type=int, default=128)
    args = parser.parse_args()

    # load ground-truth envmap 
    filename = os.path.abspath(args.envmap_path)
    gt_envmap = imageio.imread(filename)[:,:,:3]
    gt_envmap = cv2.resize(gt_envmap, (512, 256), interpolation=cv2.INTER_AREA)
    gt_envmap = torch.from_numpy(gt_envmap).cuda()
    H, W = gt_envmap.shape[:2]

    out_dir = filename[:-4]
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    assert (os.path.isdir(out_dir))

    numLgtSGs = args.num_sg
    lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., 3:4] *= 100.
    lgtSGs.requires_grad = True

    optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)

    # reload sg parameters if exists
    pretrained_file = os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs))
    if os.path.isfile(pretrained_file):
        print('Loading: ', pretrained_file)
        lgtSGs.data.copy_(torch.from_numpy(np.load(pretrained_file)).cuda())

    N_iter = 100000
    for step in range(N_iter):
        optimizer.zero_grad()
        env_map = compute_envmap(lgtSGs, H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward() 
        optimizer.step()

        if step % 50 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))

        if step % 100 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
            im = np.clip(np.power(im, 1./2.2), 0., 1.)
            im = np.uint8(im * 255.)
            imageio.imwrite(os.path.join(out_dir, 'log_im_{}.png'.format(numLgtSGs)), im)

            np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())