import sys
sys.path.append('core')
sys.path.append('Evaluator')
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import time
from frame_utils import *
# from utils import *
# from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import math
import json
import matplotlib.pyplot as plt


cudnn.benchmark = True

# read intrinsics and extrinsics
def read_camera_parameters(filename,scale,index,flag):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size

    intrinsics[:2, :] *= scale

    if (flag==0):
        intrinsics[0,2]-=index
    else:
        intrinsics[1,2]-=index
  
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

def downsampled_pair(pair_data, k, nf):
    
    pair_dict = {ref: src for ref, src in pair_data}
    pair_list = []

    for ref in pair_dict:
        l = pair_dict[ref].copy()
        head = 0
        level = 0
        while len(l) < nf:
            if head == len(l):
                head = 0
                level += 1
            assert(len(pair_dict[l[head]]) > level)
            new_f = pair_dict[l[head]][level]
            if not new_f in l and new_f != ref:
                l.append(new_f)
            head += 1
        pair_list.append((ref, l))
    pair_list.sort(key=lambda x: x[0])

    pair_list = [(ref, list(set([x // 3 * 3 for x in src if x // 3 * 3 != ref]))) for ref, src in pair_list if ref % k == 0]
    for ref, src in pair_list:
        if len(src) > 10:
            src[:] = src[:10]
    # pair_list = []
    # for ref in pair_dict:
    #     l = pair_dict[ref].copy()
    #     head = 0
    #     level = 0
    #     while len(l) < nf:
    #         if head == len(l):
    #             head = 0
    #             level += 1
    #         if len(pair_dict[l[head]]) <= level:
    #             head += 1
    #             continue
    #         new_f = pair_dict[l[head]][level]
    #         if not new_f in l and new_f != ref:
    #             l.append(new_f)
    #         head += 1
    #     pair_list.append((ref, l))
    # pair_list.sort(key=lambda x: x[0])
    return pair_list



def read_score_file(filename):
    data=[]
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            scores = [float(x) for x in f.readline().rstrip().split()[2::2]]
            data.append(scores)
    return data



# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=4.4, thre2=1430.
                                ):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    # print(np.mean(depth_ref))
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/thre1, relative_depth_diff < i/thre2) # 4 1300 4.4 1430 5 1625
        # mask = np.logical_and(dist < i/thre1, depth_diff < i/thre2)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff


def filter_depth(scan_folder, out_folder, plyfilename, photo_threshold, thre1=4.4, thre2=1430., variant_thresholding=False, vt_a=0, vt_b=0, time_downsample=1):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    score_data = read_score_file(pair_file)

    nviews = len(pair_data)
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    # for each reference view and the corresponding source views
    ct2 = -1

    pair_data = pair_data[::time_downsample] #downsampled_pair(pair_data, time_downsample, 30)

    print(pair_data)
    # assert(0)

    _, canonical_extrinsics = read_camera_parameters(os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(0)),args.scale,0,0)

    for ref_view, src_views in pair_data:

        ct2 += 1


        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depths/{:0>8}.pfm'.format(ref_view)))[0]
        h, w = ref_depth_est.shape
        ref_depth_est = cv2.resize(ref_depth_est, (int(w * args.scale), int(h * args.scale)))


        # ref_img=cv2.pyrUp(ref_img)

        #ref_depth_est=cv2.pyrUp(ref_depth_est)
        # ref_depth_est=cv2.pyrUp(ref_depth_est)

        # load the photometric mask of the reference view
        # confidence = read_pfm(os.path.join(out_folder, 'confidence_0/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = ref_depth_est

        scale=float(confidence.shape[0])/ref_img.shape[0]
        index=int((int(ref_img.shape[1]*scale)-confidence.shape[1])/2)
        flag=0
        if (confidence.shape[1]/ref_img.shape[1]>scale):
            scale=float(confidence.shape[1])/ref_img.shape[1]
            index=int((int(ref_img.shape[0]*scale)-confidence.shape[0])/2)
            flag=1

        #confidence=cv2.pyrUp(confidence)
        ref_img = cv2.resize(ref_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # ref_img=cv2.resize(ref_img,(int(ref_img.shape[1]*scale),int(ref_img.shape[0]*scale)))
        # print(ref_img.shape)
        if flag == 0:
            index = int(math.ceil((ref_img.shape[1] - confidence.shape[1]) / 2))
        else:
            index = int(math.ceil((ref_img.shape[0] - confidence.shape[0]) / 2))

        if (flag==0):
            print(index, confidence.shape[1] + index)
            ref_img=ref_img[:,index:confidence.shape[1] + index,:]
            # ref_img=ref_img[:,index:ref_img.shape[1]-index,:]
            # assert(0)
        else:
            ref_img=ref_img[index:ref_img.shape[0]-index,:,:]

        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)),scale,index,flag)
        ref_extrinsics = ref_extrinsics @ np.linalg.inv(canonical_extrinsics)

        photo_mask = np.ones_like(confidence, dtype=bool)
        

        # photo_mask = confidence>=0

        # photo_mask = confidence > confidence.mean()

        # ref_depth_est=ref_depth_est * photo_mask


        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        # compute the geometric mask
        geo_mask_sum = 0
        geo_mask_sums=[]
        n=1
        for src_view in src_views:
          n+=1
        ct = 0

        relative_depth_diff_list = []

        # ref_nv = read_pfm(os.path.join(out_folder, 'nv/{:0>8}.pfm'.format(ref_view)))[0]
        #ref_plane = np.sum(cv2.resize(read_img(os.path.join(out_folder, 'planes/{:0>8}.png'.format(ref_view))), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR), axis=-1)


        if variant_thresholding:
            ref_plane = np.sum(cv2.resize(read_img(os.path.join(out_folder, 'planes/{:0>8}.png'.format(ref_view))), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR), axis=-1)
			# thre1 = 4 * (vt_a ** ref_nv * vt_b)
            # thre2 = 1300 * (vt_a ** ref_nv * vt_b)
            # thre1 = 4 * np.where(ref_nv < 15, vt_a, vt_b)
            # thre2 = 1300 * np.where(ref_nv < 15, vt_a, vt_b)
            thre1 = 4 * np.where(ref_plane != 0, vt_a, vt_b)
            thre2 = 1300 * np.where(ref_plane != 0, vt_a, vt_b)

        for src_view in src_views:
                ct = ct + 1
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)),scale,index,flag)
                src_extrinsics = src_extrinsics @ np.linalg.inv(canonical_extrinsics)
                
                # the estimated depth of the source view
                src_depth_est = read_pfm(os.path.join(out_folder, 'depths/{:0>8}.pfm'.format(src_view)))[0]
                h, w = src_depth_est.shape
                src_depth_est = cv2.resize(src_depth_est, (int(w * args.scale), int(h * args.scale)))


                #src_depth_est=cv2.pyrUp(src_depth_est)
                # src_depth_est=cv2.pyrUp(src_depth_est)

                # src_confidence = read_pfm(os.path.join(out_folder, 'confidence_0/{:0>8}.pfm'.format(src_view)))[0]

                # src_mask=src_confidence>0.1
                # src_mask=src_confidence>src_confidence.mean()

                # src_depth_est=src_depth_est*src_mask



                masks, geo_mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                            ref_extrinsics,
                                                                                            src_depth_est,
                                                                                            src_intrinsics, src_extrinsics,
                                                                                            thre1, thre2)

                relative_depth_diff_list.append(relative_depth_diff)

                if (ct==1):
                    for i in range(2,n):
                        geo_mask_sums.append(masks[i-2].astype(np.int32))
                else :
                    for i in range(2,n):
                        geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

                geo_mask_sum+=geo_mask.astype(np.int32)

                all_srcview_depth_ests.append(depth_reprojected)

                # all_srcview_x.append(x2d_src)
                # all_srcview_y.append(y2d_src)
                # all_srcview_geomask.append(geo_mask)


        # relative_depth_diff = np.stack(relative_depth_diff_list)
        # print(relative_depth_diff.shape)
        # mean_depth_diff = np.mean(relative_depth_diff, 0).reshape(-1)
        # plt.hist(mean_depth_diff, bins=40, range=(0, 0.05))
        # plt.savefig("hist.png")
        # if ref_view == 3:
        #     assert(0)

        geo_mask=geo_mask_sum>=n

        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)
            print(geo_mask.mean())

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)


        if (not isinstance(geo_mask, bool)):

            final_mask = np.logical_and(photo_mask, geo_mask)

            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)

            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                        photo_mask.mean(),
                                                                                        geo_mask.mean(),
                                                                                        final_mask.mean()))



            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
            valid_points = final_mask
            print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]  # hardcoded for DTU dataset
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))

            # # set used_mask[ref_view]
            # used_mask[ref_view][...] = True
            # for idx, src_view in enumerate(src_views):
            #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
            #     src_y = all_srcview_y[idx].astype(np.int)
            #     src_x = all_srcview_x[idx].astype(np.int)
            #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0) # N x 3
    vertex_colors = np.concatenate(vertex_colors, axis=0) # N x 3

    vertex_valid = np.logical_and(vertexs<1000., vertexs>-1000.) # N x 3
    vertex_valid = np.all(vertex_valid, axis=1) # N

    vertexs = vertexs[vertex_valid]
    vertex_colors = vertex_colors[vertex_valid]

    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    # save_depth()




    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')

    # parser.add_argument('--testpath', default='/data1/wzz/tnt/',help='testing data path')
    parser.add_argument('--setting', type=str)
    parser.add_argument('--testing', type=int, default=False)
    parser.add_argument('--variant_thresholding', type=int, default=False)
    parser.add_argument('--vt_a', type=float, default=0)
    parser.add_argument('--vt_b', type=float, default=0)
    parser.add_argument('--scan', type=str)
    parser.add_argument('--outdir', default='/data1/wzz/outputs_tnt_1101', help='output dir')
    parser.add_argument('--display', action='store_true', help='display depth images and masks')
    parser.add_argument('--thre1', type=float, default=4.4)
    parser.add_argument('--thre2', type=float, default=1430)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--time_downsample', type=int, default=1)

    # parse arguments and check
    args = parser.parse_args()
    # print("argv:", sys.argv[1:])
    # print_args(args)


    with open('dir.json') as f:
        d = json.load(f)
    d = d[args.setting]

    if not args.testing:
        args.testpath = d["validating_dir"]
    else:
        args.testpath = d["testing_dir"]

        
    scan = args.scan
    scan_folder = os.path.join(args.testpath, scan)
    out_folder = os.path.join(args.outdir, scan)
    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    # if (args.test_dataset=='dtu'):
    #     scan_id = int(scan[4:])
    #     photo_threshold=0.35
    #     filter_depth(scan_folder, out_folder, os.path.join(args.outdir, 'mvsnet_{:0>3}_l3.ply'.format(scan_id) ), photo_threshold)
    # if (args.test_dataset=='tanks'):
    photo_threshold=-1e9 #0.3
    filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), photo_threshold, args.thre1, args.thre2, args.variant_thresholding, args.vt_a, args.vt_b, args.time_downsample)
