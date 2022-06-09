from __future__ import print_function, division
import sys

sys.path.append('..')
sys.path.append('../core')
sys.path.append('../datasets')

import argparse
import os
import cv2
import time
import numpy as np
import subprocess
import json
import glob
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from raft import *
from llff import LLFF, LLFFTest
import projective_ops as pops
from frame_utils import write_pfm, save_ply
from geom_utils import PtsUnprojector, check_depth_consistency

from plyfile import PlyData, PlyElement

EPS = 1e-2

def metric(disp_est, disp_gt, disp_clamp=1e-4):
    valid = disp_gt > 0.0
    ht, wd = disp_gt.shape[-2:]
    disp_est_resized = F.interpolate(disp_est, [ht, wd], mode='bilinear', align_corners=True)
    epe = (1.0 / disp_est_resized.clamp(min=disp_clamp) - 1.0 / disp_gt.clamp(min=disp_clamp)).abs()
    epe = epe.view(-1)[valid.view(-1)]
    return epe.mean().item()


def inference(args):
    params = {
        "corr_len": 2 if args.pooltype in ['maxmean', "meanvar"] else 1,
        "inference": 1,
        "HRv2": 0,
        "invariance": 0,
    }

    for k in list(vars(args).keys()):
        params[k] = vars(args)[k]

    model = RAFT(**params)

    model.cuda()

    model.eval()

    if args.ckpt is not None:
        tmp = torch.load(args.ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=False)

    with open('dir.json') as f:
        d = json.load(f)
    d = d[args.setting]
    datasetname = d["dataset"]

    gpuargs = {'num_workers': 4, 'drop_last': False, 'shuffle': False, 'pin_memory': True}

    dataset_params = {}
    if datasetname == "Blended":
        dataset_params["scaling"] = args.scaling

    if datasetname == "DTU":
        dataset_params["pairs_provided"] = args.pairs_provided
        if args.training:
            dataset_params["crop_size"] = [1200, 1600]
            dataset_params["resize"] = [1200, 1600]
            dataset_params["light_number"] = -1
        else:
            dataset_params["start"] = args.start
            dataset_params["end"] = args.end
            dataset_params['source_views'] = np.arange(49)[np.mod(np.arange(49), 7) != 2]

    if datasetname == "LLFF":
        total_num_views = len(sorted(glob.glob(os.path.join(args.scan_path, "DTU_format", "images", "*.jpg"))))
        indicies = np.arange(total_num_views)
        # dataset_params["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 5) != 2])

    # dataset = LLFF(args.scan_path, args.num_frames, data_augmentation=False)

    if args.training:
        dataset = eval(datasetname)(args.scan_path, args.num_frames, **dataset_params)
    else:
        dataset = eval(datasetname + 'Test')(args.scan_path, args.num_frames, size_operations=args.size_operations,
                                              **dataset_params)

    loader = DataLoader(dataset, batch_size=1, **gpuargs)
    scene_name = args.scan_path.split("/")[-1]

    # print(scene_name)
    tgt_size = (int(args.scale*150), int(args.scale*200))
    unprojector = PtsUnprojector().cuda()

    with torch.no_grad():

        xyzs_est = []
        xyzs_gt = []
        rgbs_est = []
        rgbs_gt = []

        all_rgb_gts = []
        all_poses = []
        all_intrinsics = []
        all_pred_depths = []
        all_gt_depths = []

        for i, data_blob in enumerate(loader):
            print(i)
            if i < args.start or i >= args.end: continue
            scale = None
            # try:
            #     images, poses, intrinsics = data_blob
            # except:
            if datasetname == "LLFF":
                images, depths, poses, intrinsics, scale = data_blob
                out_path = os.path.join(f"{args.output_folder}/%08d.pfm" % (i+1))
                depth_vis_path = os.path.join(args.output_folder, "depth_visual_%08d.png" % (i+1))

            else:
                if args.training:
                    images, _, poses, intrinsics = data_blob
                    light_id, scene_id, frame_id = dataset.get_blob_info(i)
                    subprocess.call("mkdir -p %s" % os.path.join(args.output_folder, scene_id), shell=True)
                    out_path = os.path.join(args.output_folder, scene_id, "depth_map_%04d_%d.pfm" % (frame_id, light_id))
                    # depth_vis_path = cv2.imwrite(os.path.join(args.output_folder, "depth_visual_%08d.png" % (i+1)), disp_to_vis)
                    print(out_path)
                else:
                    images, poses, intrinsics, indices, scale = data_blob
                    out_path = os.path.join(f"{args.output_folder}/%08d.pfm" % indices[0])
                    depth_vis_path = os.path.join(args.output_folder, "depth_visual_%08d.png" % indices[0])

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()

            graph = OrderedDict()
            graph[0] = [i for i in range(1, images.shape[1])]

            # if args.init_path != "":
            #     path = os.path.join(args.init_path, scene_name, "%d.npy" % indices[0])
            #     disp_est = model(images, poses, intrinsics, graph,
            #         initial_disp=path, initial_scale=scale.item()
            #     )
            # else:
            #     # mem_report()
            disp_est = model(images, poses, intrinsics, graph)

            disp_est = disp_est[-1]

            if not scale is None:
                disp_est *= scale.cuda()

            disp_est = disp_est.clamp(min=1e-4)

            depth_est = torch.where(disp_est > 0, 1.0 / disp_est, torch.zeros_like(disp_est))


            factor = 8 / args.scale
            intrinsics_scaled = intrinsics[:, 0]  # B x 4 x 4
            intrinsics_scaled[:, 0] /= factor
            intrinsics_scaled[:, 1] /= factor

            all_poses.append(poses[:, 0])
            all_intrinsics.append(intrinsics_scaled)
            all_pred_depths.append(depth_est)

            rgb_gt = F.interpolate(images[:, 0], tgt_size, mode='bilinear', align_corners=True)
            rgb_gt = rgb_gt * 2.0 / 255.0 - 1.0
            all_rgb_gts.append(rgb_gt)

            if datasetname == "LLFF":
                depths = depths.cuda()
                depths = depths[:, [0]]
                disp_gt = torch.where(depths > 0, 1.0 / depths, torch.zeros_like(depths))

                epe = metric(disp_est, disp_gt)
                print('epe: %.2f' % epe)

            depth_gt_scaled = F.interpolate(depths, tgt_size, mode='nearest')
            all_gt_depths.append(depth_gt_scaled)

            tosave = disp_est.cpu().numpy()
            # np.save(os.path.join(args.output_folder, "%d.npy" % i), tosave)
            tosave = tosave[0, 0]
            # x = np.where(tosave == 0, 0, 1 / tosave).astype(np.float32)

            max_disp = 1. / 400.
            disp_to_vis = (tosave / max_disp * 255.0).astype(np.uint8)



            if args.training:
                cv2.imwrite(os.path.join(args.output_folder, scene_id, "depth_visual_%04d_%d.png" % (frame_id, light_id)), disp_to_vis)
            else:
                cv2.imwrite(depth_vis_path, disp_to_vis)

        all_poses = torch.cat(all_poses)
        all_intrinsics = torch.cat(all_intrinsics)
        all_pred_depths = torch.cat(all_pred_depths)
        all_gt_depths = torch.cat(all_gt_depths)
        all_rgb_gts = torch.cat(all_rgb_gts)

        depth_masks = check_depth_consistency(all_pred_depths, all_poses, all_intrinsics)
        depth_masks_gt = check_depth_consistency(all_gt_depths, all_poses, all_intrinsics)

        for i in range(all_pred_depths.shape[0]):
            tosave = torch.clone(all_pred_depths[i:i+1]) # 1 x 1 x H x W

            # we move the post-processing into the view-syn model, so no masking here!
            # tosave[depth_masks[i:i+1] < 0.5] = -1.0 # mask out
            
            tosave = tosave.cpu().numpy()
            # np.save(os.path.join(args.output_folder, "%d.npy" % i), tosave)
            x = tosave[0, 0]

            if hasattr(dataset, "depth_scale"):
                # scale back before saving for LLFF dataset
                x = x * dataset.depth_scale
                # print(dataset.depth_scale)

            # write_pfm(out_path, x)
            write_pfm(os.path.join(f"{args.output_folder}/%08d.pfm" % (i+1)), x)

        for i in range(all_pred_depths.shape[0]):
            xyz_est = unprojector(all_pred_depths[i:i+1], all_poses[i:i+1], all_intrinsics[i:i+1], mask=depth_masks[i:i+1])  # N x 3
            xyz_gt = unprojector(all_gt_depths[i:i+1], all_poses[i:i+1], all_intrinsics[i:i+1], mask=depth_masks_gt[i:i+1])  # N x 3

            colors_est = unprojector.apply_mask(all_rgb_gts[i:i+1], mask=depth_masks[i:i+1])  # N x 3
            colors_gt = unprojector.apply_mask(all_rgb_gts[i:i + 1], mask=depth_masks_gt[i:i + 1])  # N x 3

            xyzs_est.append(xyz_est)
            xyzs_gt.append(xyz_gt)
            rgbs_est.append(colors_est)
            rgbs_gt.append(colors_gt)

        xyzs_est = torch.cat(xyzs_est)
        xyzs_gt = torch.cat(xyzs_gt)
        rgbs_est = torch.cat(rgbs_est)
        rgbs_gt = torch.cat(rgbs_gt)

        save_ply(os.path.join(args.output_folder, "pctld.ply"), xyzs_est, rgbs_est)
        save_ply(os.path.join(args.output_folder, "pctld_gt.ply"), xyzs_gt, rgbs_gt)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--memory_file', type=str, default=None)

    '''datasets params'''
    parser.add_argument('--setting', type=str, default='DTU')
    parser.add_argument('--single_scan', type=str, default='')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--testing', type=int, default=False)
    parser.add_argument('--training', type=int, default=False)
    parser.add_argument('--output_folder', help='location to write restults')
    parser.add_argument('--scale', type=float, default=2)
    # parser.add_argument('--fusion_scale', type=float, default=1)
    parser.add_argument('--scaling', type=str, default='median')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=99999999)
    parser.add_argument('--pairs_provided', type=int, default=False)

    '''checkpoints and model params'''
    parser.add_argument('--ckpt', help='model weights')
    parser.add_argument('--DD', type=int, default=128)
    parser.add_argument('--Dnear', type=float, default=.0025)
    parser.add_argument('--Dfar', type=float, default=.0)
    parser.add_argument('--HR', type=int, default=False)
    parser.add_argument('--HRv2', type=int, default=False)
    parser.add_argument('--kernel_z', type=int, default=3)
    parser.add_argument('--kernel_r', type=int, default=3)
    parser.add_argument('--kernel_q', type=int, default=3)
    parser.add_argument('--kernel_corr', type=int, default=3)
    parser.add_argument('--dim0_corr', type=int, default=128)
    parser.add_argument('--dim1_corr', type=int, default=128)
    parser.add_argument('--dim_net', type=int, default=128)
    parser.add_argument('--dim_inp', type=int, default=128)
    parser.add_argument('--dim0_delta', type=int, default=256)
    parser.add_argument('--kernel0_delta', type=int, default=3)
    parser.add_argument('--kernel1_delta', type=int, default=3)
    parser.add_argument('--dim0_upmask', type=int, default=256)
    parser.add_argument('--kernel_upmask', type=int, default=3)
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--s_disp_enc', type=int, default=7)
    parser.add_argument('--num_iters', type=int, default=16)
    parser.add_argument('--dim_fmap', type=int, default=128)
    parser.add_argument('--cascade', type=int, default=False)
    parser.add_argument('--cascade_v2', type=int, default=False)
    parser.add_argument('--num_iters1', type=int, default=8)
    parser.add_argument('--num_iters2', type=int, default=5)
    parser.add_argument('--DD_fine', type=int, default=320)
    # parser.add_argument('--len_dyna', type=int, default=44)
    parser.add_argument('--slant', type=int, default=False)
    parser.add_argument('--no_upsample', type=int, default=False)
    parser.add_argument('--pooltype', type=str, default="maxmean")
    parser.add_argument('--merge', type=int, default=0)
    parser.add_argument('--merge_permute', type=int, default=0)
    parser.add_argument('--visibility', type=int, default=False)
    parser.add_argument('--visibility_v2', type=int, default=False)

    parser.add_argument('--output_appearance_features', type=int, default=False)
    parser.add_argument('--appearace_iters', type=int, default=3)
    parser.add_argument('--dim_appearace', type=int, default=256)

    args = parser.parse_args()
    args.len_dyna = (2 * args.radius + 1) * 2 ** (args.num_levels - 1)

    subprocess.call(f"rm -f {args.memory_file}", shell=True)

    output_folder = args.output_folder

    with open('dir.json') as f:
        d = json.load(f)
    d = d[args.setting]

    assert(not (args.testing and args.training)) # can't do both

    dir = os.path.join(d["testing_dir"])

    if args.training:
        args.scan_path = dir
        args.output_folder = os.path.join(output_folder, "train_depths")
        subprocess.call("mkdir -p %s" % args.output_folder, shell=True)
        args.vis = 0
        args.size_operations = [
            ("scale", args.scale)
        ]
        inference(args)

    else:
        for scan in os.listdir(dir):
            if scan[-4:] == ".zip": continue
            if args.single_scan != '' and scan != args.single_scan: continue
            subprocess.call("mkdir -p %s" % os.path.join(output_folder, scan, "depths"), shell=True)
            args.scan_path = os.path.join(dir, scan)
            args.output_folder = os.path.join(output_folder, scan, "depths")
            args.vis = 0
            # args.scanname = scan
            args.size_operations = [
                ("scale", args.scale)
            ]
            inference(args)
            print(scan)