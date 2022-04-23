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

from PIL import Image

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


MAX_FLOW = 400


@torch.no_grad()
def create_sintel_submission(model, warm_start=False, fixed_point_reuse=False, output_path='sintel_submission', **kwargs):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        sequence_prev, flow_prev, fixed_point = None, None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
                fixed_point = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            
            flow_low, flow_pr, info = model(image1, image2, flow_init=flow_prev, cached_result=fixed_point, **kwargs)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            
            # You may choose to use some hacks here,
            # for example, warm start, i.e., reusing the f* part with a borderline check (forward_interpolate),
            # which was orignally taken by RAFT.
            # This trick usually (only) improves the optical flow estimation on the ``ambush_1'' sequence,
            # in terms of clearer background estimation.
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            # Note that the fixed point reuse usually does not improve performance.
            # It facilitates the convergence.
            # To improve performance, the borderline check like ``forward_interpolate'' is necessary.
            if fixed_point_reuse:
                net, flow_pred_low = info['cached_result']
                flow_pred_low = forward_interpolate(flow_pred_low[0])[None].cuda()
                fixed_point = (net, flow_pred_low)

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the KITTI leaderboard """
    model.eval()
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
        frame_utils.writeFlowKITTI(output_filename, flow)
        

@torch.no_grad()
def validate_chairs(model, **kwargs):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    rho_list = []
    best = kwargs.get("best", {"epe":1e8})

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr, info = model(image1, image2, **kwargs)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        rho_list.append(info['sradius'].mean().item())

    epe = np.mean(np.concatenate(epe_list))
    best['epe'] = min(epe, best['epe'])
    print(f"Validation Chairs EPE: {epe} ({best['epe'] if best['epe'] < 1e8 else 'N/A'})")

    if np.mean(rho_list) != 0:
        print("Spectral radius: %f" % np.mean(rho_list))

    return {'chairs': epe}


@torch.no_grad()
def validate_things(model, **kwargs):
    """ Peform validation using the FlyingThings3D (test) split """
    model.eval()
    results = {}
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(split='test', dstype=dstype)
        epe_list = []
        epe_w_mask_list = []
        rho_list = []
        
        print(f'{dstype} length', len(val_dataset))
        
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, info = model(image1, image2, **kwargs)
            flow = padder.unpad(flow_pr[0]).cpu()
            
            # exlude invalid pixels and extremely large diplacements
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            valid = (valid >= 0.5) & (mag < MAX_FLOW)
            
            loss = (flow - flow_gt)**2
            
            if torch.any(torch.isnan(loss)):
                print(f'Bad prediction, {val_id}')
            
            loss_w_mask = valid[None, :] * loss
            
            if torch.any(torch.isnan(loss_w_mask)):
                print(f'Bad prediction after mask, {val_id}')
                print('Bad pixels num', torch.isnan(loss).sum())
                print('Bad pixels num after mask', torch.isnan(loss_w_mask).sum())
                continue

            epe = torch.sum(loss, dim=0).sqrt()
            epe_w_mask = torch.sum(loss_w_mask, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            epe_w_mask_list.append(epe_w_mask.view(-1).numpy())
            rho_list.append(info['sradius'].mean().item())
            
            if (val_id + 1) % 100 == 0:
                print('EPE', np.mean(epe_list), 'EPE w/ mask', np.mean(epe_w_mask_list))

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        
        epe_all_w_mask = np.concatenate(epe_w_mask_list)
        epe_w_mask = np.mean(epe_all_w_mask)
        px1_w_mask = np.mean(epe_all_w_mask<1)
        px3_w_mask = np.mean(epe_all_w_mask<3)
        px5_w_mask = np.mean(epe_all_w_mask<5)

        print("Validation         (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Validation w/ mask (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe_w_mask, px1_w_mask, px3_w_mask, px5_w_mask))
        results[dstype] = np.mean(epe_list)
        results[dstype+'_w_mask'] = np.mean(epe_w_mask_list)
        
        if np.mean(rho_list) != 0:
            print("Spectral radius (%s): %f" % (dstype, np.mean(rho_list)))

    return results


@torch.no_grad()
def validate_sintel(model, **kwargs):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    best = kwargs.get("best", {"clean-epe":1e8, "final-epe":1e8})
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        rho_list = []
        info = {"sradius": None, "cached_result": None}

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, info = model(image1, image2, **kwargs)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            rho_list.append(info['sradius'].mean().item())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        best[dstype+'-epe'] = min(epe, best[dstype+'-epe'])
        print(f"Validation ({dstype}) EPE: {epe} ({best[dstype+'-epe'] if best[dstype+'-epe'] < 1e8 else 'N/A'}), 1px: {px1}, 3px: {px3}, 5px: {px5}")
        results[dstype] = np.mean(epe_list)

        if np.mean(rho_list) != 0:
            print("Spectral radius (%s): %f" % (dstype, np.mean(rho_list)))

    return results


@torch.no_grad()
def validate_kitti(model, **kwargs):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    best = kwargs.get("best", {"epe":1e8, "f1":1e8})
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list, rho_list = [], [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr, info = model(image1, image2, **kwargs)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        rho_list.append(info['sradius'].mean().item())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    best['epe'] = min(epe, best['epe'])
    best['f1'] = min(f1, best['f1'])
    print(f"Validation KITTI: EPE: {epe} ({best['epe'] if best['epe'] < 1e8 else 'N/A'}), F1: {f1} ({best['f1'] if best['f1'] < 1e8 else 'N/A'})")
    
    if np.mean(rho_list) != 0:
        print("Spectral radius %f" % np.mean(rho_list))

    return {'kitti-epe': epe, 'kitti-f1': f1}


