"""
Usage:
python -m demo.demo_hoi -e xxx  --image xxx.png [--weight ....ckpt] [--out ]

Usage:
    # Run on image for bicycle class.
    python demo.py --filename my_image.jpg --class_name bicycle
    # Run on image without coarse interaction loss.
    python demo.py --filename my_image.jpg --class_name bicycle --lw_inter 0
    # Run on image with higher weight on depth ordering loss.
    python demo.py --filename my_image.jpg --class_name bicycle --lw_depth 1.
"""
import argparse
import logging
import os.path as osp

import numpy as np

import sys
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from datasets import build_dataloader
from nnutils.hand_utils import ManopthWrapper
from nnutils.hoiapi import get_hoi_predictor
from nnutils import mesh_utils, image_utils

def get_args():
    parser = argparse.ArgumentParser(description="Optimize object meshes w.r.t. human.")
    parser.add_argument(
        "--filename", default="demo/test.jpg", help="Path to image."
    )
    parser.add_argument("--out", default="/checkpoint/yufeiy2/hoi_output/reproduce/demo/", help="Dir to save output.")

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='/checkpoint/yufeiy2/hoi_output/release_model/obman'
    )
    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):
    hoi_predictor = get_hoi_predictor(args)
    # hoi_predictor.model.cfg.DB.NAME = 'ho3d_vid'
    # dataloader = build_dataloader(hoi_predictor.model.cfg, 'train', False, False)

    hoi_predictor.model.cfg.DB.NAME = 'obman'
    hoi_predictor.model.cfg.MODEL.BATCH_SIZE = 1
    dataloader = build_dataloader(hoi_predictor.model.cfg, 'test', False, False)

    for i, data in enumerate(dataloader):
        if i < 14 * 8:
            continue
        output = hoi_predictor.forward_to_mesh(data)
        vis_hand_object(output, data, None, args.out + '/test_%d' % i)
        print(i)
        if i > 14 * 10:
            break


def vis_hand_object(output, data, image, save_dir):
    hHand = output['hHand']
    hObj = output['hObj']
    device = hObj.device

    cam_f, cam_p = data['cam_f'], data['cam_p']
    cTh = data['cTh']

    hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
    cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, out_size=512)
    image_utils.save_gif(image_list, save_dir + '_cHoi')
    
    oMesh = data['mesh'].to(device)
    oMesh.textures = mesh_utils.pad_texture(oMesh)
    hMesh = mesh_utils.apply_transform(oMesh, data['hTo'].to(device))
    hHoi = mesh_utils.join_scene([hMesh, hHand]).to(device)
    cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
    image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, out_size=512)
    image_utils.save_gif(image_list, save_dir + '_gt')

    image_utils.save_images(data['image'], save_dir + '_inp', scale=True)

    # iHoi = mesh_utils.render_mesh(cHoi, cameras, out_size=512)
    # image = F.interpolate(data['image'], 512)
    # image_utils.save_images(image, save_dir + '_inp', scale=True)
    
    # image_utils.save_images(iHoi['image'], save_dir + '_cHoi', bg=image/2+0.5, mask=iHoi['mask'])
    # image_list = mesh_utils.render_geom_rot(hHoi, 'el', True, out_size=512)
    # image_utils.save_images(image_list[15], save_dir + '_hHoi')





if __name__ == "__main__":
    main(get_args())
