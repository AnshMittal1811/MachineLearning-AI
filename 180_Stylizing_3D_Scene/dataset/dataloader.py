import os
import numpy as np
import logging

from dataset.data_util import *
logger = logging.getLogger(__package__)

# Resize all images of the scene to the same size
image_size_table ={
    "tat_intermediate_Playground": [548, 1008],
    "tat_intermediate_Family": [1084, 1957],
    "tat_intermediate_Francis": [1086, 1959],
    "tat_intermediate_Horse": [1084, 1958],
    "tat_training_Truck": [546, 980]
}

def load_data_split(basedir, scene, split, try_load_min_depth=True, only_img_files=False, seed=None):
    """
    :param split train | validation | test
    """
    scenes = sorted(os.listdir(basedir))
    all_ray_samplers = []
    
    scene_dir = os.path.join(basedir, scene, split)
    
    # camera parameters files
    intrinsics_files = find_files(os.path.join(scene_dir, "intrinsics"), exts=['*.txt'])
    pose_files = find_files(os.path.join(scene_dir, "pose"), exts=['*.txt'])
    img_files = find_files(os.path.join(scene_dir, "rgb"), exts=['*.png', '*.jpg'])
    
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))
    logger.info('raw img_files: {}'.format(len(img_files)))

    cam_cnt = len(pose_files)
    logger.info("Dataset len is {}".format(cam_cnt))
    assert(len(img_files) == cam_cnt)

    # img files
    style_dir = os.path.join("./wikiart", split)
    style_img_files = find_files(style_dir, exts=['*.png', '*.jpg'])
    logger.info("Number of style images is {}".format(len(style_img_files)))
    
    # create ray samplers
    ray_samplers = []
    H, W = image_size_table[scene]
    
    if seed != None:
        np.random.seed(seed)

    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                img_path = img_files[i],
                                                mask_path=None,
                                                min_depth_path=None,
                                                max_depth=None,
                                                style_imgs = style_img_files
                                                ))
        
    logger.info('Split {}, # views: {}'.format(split, cam_cnt))
    
    return ray_samplers
