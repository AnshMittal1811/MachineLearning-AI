import sys

sys.path.append('core')

import argparse
import copy
import os
import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import cv2
from PIL import Image

from utils import flow_viz, frame_utils
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def sintel_visualization(model, split='test', warm_start=False, fixed_point_reuse=False, output_path='sintel_viz', **kwargs):
    """ Create visualization for the Sintel dataset """
    model.eval()
    for dstype in ['clean', 'final']:
        split = 'test' if split == 'test' else 'training'
        test_dataset = datasets.MpiSintel(split=split, aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev, fixed_point = None, None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
                fixed_point = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            
            flow_low, flow_pr, info = model(image1, image2, flow_init=flow_prev, cached_result=fixed_point, **kwargs)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            if fixed_point_reuse:
                net, flow_pred_low = info['cached_result']
                flow_pred_low = forward_interpolate(flow_pred_low[0])[None].cuda()
                fixed_point = (net, flow_pred_low)

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.png' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # visualizaion
            img_flow = flow_viz.flow_to_image(flow)
            img_flow = cv2.cvtColor(img_flow, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_file, img_flow, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

            sequence_prev = sequence


@torch.no_grad()
def kitti_visualization(model, split='test', output_path='kitti_viz'):
    """ Create visualization for the KITTI dataset """
    model.eval()
    split = 'testing' if split == 'test' else 'training'
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr, _ = model(image1, image2)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        
        # visualizaion
        img_flow = flow_viz.flow_to_image(flow)
        img_flow = cv2.cvtColor(img_flow, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_filename, img_flow, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


