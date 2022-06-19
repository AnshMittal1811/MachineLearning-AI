import cv2
import numpy as np
import scipy
import scipy.io as scio
from PIL import Image
import time
import math
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import torch
def get_density_map_gaussian(H, W, ratio_h, ratio_w,  points, adaptive_kernel=False, fixed_value=15):
    h = H
    w = W
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
        sigma = fixed_value
        sigma = max(1, sigma)

        gaussian_radius = 7
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]

    return density_map

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# SHA: 300, 182
# SHB: 400, 316
if __name__ == "__main__":

    is_train = 0 # 0 for test
    # The processed data should be stored in folder with names silimar with `xx/train_data/`
    train_test_for_gt_SH = 'train_data' if is_train else 'test_data'
    train_test_for_gt_type2 = 'train' if is_train else 'test'
    train_test_for_gt_type3 = 'train' if is_train else 'val'
    train_test_for_den = 'den/train' if is_train else 'den/test'
    dataset = 'QNRF_small'

    if dataset == 'SHA':
#        num_img = 300 if is_train else 182
        image_dir_path = "/home/weigang1/zx/hrnet_counting-master/data_all/ShanghaiTech/part_A_final/" + train_test_for_gt_SH + "/images"
        ground_truth_dir_path = "/home/weigang1/zx/hrnet_counting-master/data_all/ShanghaiTech/part_A_final/"+ train_test_for_gt_SH +"/ground_truth"
        output_den_path = "./SH_part_A/" + train_test_for_den
        # two extra paths
        output_img_path = "./SH_part_A/ori/" + train_test_for_gt_SH + "/images"
        output_mat_path = './SH_part_A/ori/' + train_test_for_gt_SH + "/ground_truth"
        
    elif dataset == 'SHB':
 #       num_img = 400 if is_train else 316
        image_dir_path = "/home/weigang1/zx/hrnet_counting-master/data_all/ShanghaiTech/part_B_final/" + train_test_for_gt_SH + "/images"
        ground_truth_dir_path = "/home/weigang1/zx/hrnet_counting-master/data_all/ShanghaiTech/part_B_final/" + train_test_for_gt_SH + "/ground_truth"
        output_den_path = "./SH_part_B/" + train_test_for_den
        # two extra paths
        output_img_path = "./SH_part_B/ori/" + train_test_for_gt_SH + "/images"
        output_mat_path = './SH_part_B/ori/' + train_test_for_gt_SH + "/ground_truth"
    elif dataset.find('QNRF') != -1:
  #      num_img = 1201 if is_train else 334
        image_dir_path = "/home/weigang1/zx/NDConv/data_all/UCF-QNRF_ECCV18/" + train_test_for_gt_type2 + "/images"
        ground_truth_dir_path = "/home/weigang1/zx/NDConv/data_all/UCF-QNRF_ECCV18/" + train_test_for_gt_type2 + "/ground_truth"
        output_den_path = "/home/weigang1/zx/NDConv/Processed_QNRF_large_oriImg/" + train_test_for_den

        # two extra paths
        output_img_path = "/home/weigang1/zx/NDConv/Processed_QNRF_large_oriImg/ori/" + train_test_for_gt_SH + "/images" # using `gt_SH` here
        output_mat_path = '/home/weigang1/zx/NDConv/Processed_QNRF_large_oriImg/ori/' + train_test_for_gt_SH + "/ground_truth"
#    elif dataset == 'UCF50': # take all images as testing images
#        num_img = 50
#        image_dir_path = "UCF_CC_50/images/UCF_CC_50_img"
#        ground_truth_dir_path = "UCF_CC_50/UCF_CC_50_mat"
#        output_den_path = "./UCF50/" + train_test
    elif dataset == 'NWPU_large':
        image_dir_path = "NWPU/" + train_test_for_gt_type3 + "/imgs"
        ground_truth_dir_path = "NWPU/" + train_test_for_gt_type3 + "/mats"
        output_den_path = "./Processed_NWPU_large_oriImg/" + train_test_for_den
        # two extra paths
        output_img_path = "./Processed_NWPU_large_oriImg/ori/" + train_test_for_gt_SH + "/images"
        output_mat_path = './Processed_NWPU_large_oriImg/ori/' + train_test_for_gt_SH + "/ground_truth"
    elif dataset == 'JHU_large':
        image_dir_path = "JHU_offical/" + train_test_for_gt_type2 + "/images"
        ground_truth_dir_path = "JHU_offical/gt_npz/" + train_test_for_gt_type2
        output_den_path = "./Processed_JHU_large_oriImg/" + train_test_for_den
        # two extra paths
        output_img_path = "./Processed_JHU_large_oriImg/ori/" + train_test_for_gt_SH + "/images"
        output_mat_path = './Processed_JHU_large_oriImg/ori/' + train_test_for_gt_SH + "/ground_truth"
    elif dataset == 'BDdata_large':
        assert is_train == 1
        image_dir_path = "BDdata/filter_imgs_100"
        ground_truth_dir_path = "BDdata/filter_mats_100"
        output_den_path = "./Processed_BDdata_large_oriImg/" + train_test_for_den
        # two extra paths
        output_img_path = "./Processed_BDdata_large_oriImg/ori/" + train_test_for_gt_SH + "/images"
        output_mat_path = './Processed_BDdata_large_oriImg/ori/' + train_test_for_gt_SH + "/ground_truth"
    else:
        assert 1==2
        

    mkdirs(output_den_path)
    mkdirs(output_img_path)
    mkdirs(output_mat_path)

    img_paths = None
    if dataset.find("QNRF") != -1: # as gt and images are in the same folder for QNRF dataset
       img_paths = glob.glob(image_dir_path + "/*.jpg")
    else:
        img_paths = glob.glob(image_dir_path + "/*")

    for img_path in img_paths:
        if dataset == 'SHA' or dataset == 'SHB':
            gt_path = os.path.join(ground_truth_dir_path, "GT_" + os.path.basename(img_path)[:-4] + ".mat")
        elif dataset.find('QNRF')!= -1:
            gt_path = os.path.join(ground_truth_dir_path, "GT_" + os.path.basename(img_path)[:-4] + ".mat")
            # gt_path = os.path.join(ground_truth_dir_path, os.path.basename(img_path)[:-4] + "_ann.mat")
        elif dataset == 'NWPU' or dataset == 'NWPU_large' or dataset == 'BDdata_large':
            gt_path = os.path.join(ground_truth_dir_path, os.path.basename(img_path)[:-4] + ".mat")
        elif dataset == 'JHU' or dataset == 'JHU_large':
            gt_path = os.path.join(ground_truth_dir_path, os.path.basename(img_path)[:-4] + ".npz")

        else:
            assert 1==2

        print('Processing img: ', img_path)

        img = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
    
        height = img.size[1]
        width = img.size[0]

        if dataset == 'SHA' or dataset == 'SHB':
            points = scio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        elif dataset.find('QNRF') != -1 or dataset == 'UCF50' or dataset.find('NWPU') != -1 or dataset.find('BDdata') != -1:
            points = scio.loadmat(gt_path)['annPoints']
        elif dataset.find('JHU') != -1:
            points = np.load(gt_path)['loc_info']
        else:
            assert 1==2


        resize_height = height
        resize_width = width

        if dataset == 'SHA' or dataset == 'UCF50':
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width

            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset == 'SHB':
            resize_height = 768
            resize_width = 1024
        elif dataset == 'QNRF_small':
            if resize_width >= 1024:
                tmp = resize_width
                resize_width = 1024
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 1024:
                tmp = resize_height
                resize_height = 1024
                resize_width = (resize_height / tmp) * resize_width

            if resize_height <= 512:
                tmp = resize_height
                resize_height = 512
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 512:
                tmp = resize_width
                resize_width = 512
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 1024/512:
                    resize_width = 1024
                    resize_height = 512
            else:
                if resize_height / resize_width > 1024/512:
                    resize_height = 1024
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        #
        elif dataset == 'QNRF_large' or dataset == 'NWPU_large' or dataset == 'JHU_large' or dataset == 'BDdata_large':
            if resize_width >= 2048:
                tmp = resize_width
                resize_width = 2048
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 2048:
                tmp = resize_height
                resize_height = 2048
                resize_width = (resize_height / tmp) * resize_width

            if resize_height <= 512:
                tmp = resize_height
                resize_height = 512
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 512:
                tmp = resize_width
                resize_width = 512
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 2048/512:
                    resize_width = 2048
                    resize_height = 512
            else:
                if resize_height / resize_width > 2048/512:
                    resize_height = 2048
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            assert 1==2


        ratio_h = (resize_height) / (height)
        ratio_w = (resize_width) / (width)
        gt = get_density_map_gaussian(resize_height, resize_width, ratio_h, ratio_w, points, False, 4)
        #gt = np.reshape(gt, [resize_height, resize_width])  # transpose into w, h

        # transfer gt to float16 to save storage
        #gt = gt.astype(np.float16)

        # Three stuffs to store
        # 1. images with new folders
        #cv2.imwrite(os.path.join(output_img_path, dataset + '_' + os.path.basename(img_path)), img_cv)
        os.system('copy '+ img_path + ' ./' + os.path.join(output_img_path, dataset + '_' + os.path.basename(img_path)))
        # assert 1==2
        # os.system('cp '+ img_path + ' ./' + os.path.join(output_img_path, dataset + '_' + os.path.basename(img_path)))
        # 2. save density maps
        np.save(os.path.join(output_den_path, dataset + "_" + os.path.basename(img_path)[:-4] + ".npy"), gt) # some extensions are '.JPG', so...
        # 3. save mats
        scio.savemat(os.path.join(output_mat_path, dataset + "_" + os.path.basename(img_path)[:-4] + ".mat"), {'annPoints':points})





    print("complete!")
