import os
import sys
import math
import numpy as np
import argparse

def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)   

def load_intrinsics(filepath):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        return intrinsics
    except ValueError:
        pass
    
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    
    return full_intrinsic

def convert_pose_nsvf_to_pytorch3d(transform_matrix):
    '''
    Input: transform matrix (in NSVF format)
    Output: rotation matrix & translation matrix (in Replica format)
    '''
    # Transform from camera2world to world2camera
    transform_matrix = np.linalg.inv(transform_matrix).astype(np.float32)
    # KiloNeRF processing
    transform_matrix[:3, 1:3] = -transform_matrix[:3, 1:3]
    # Ours (neurmips) convention
    R = transform_matrix[:3,:3]
    T = transform_matrix[:3,3]
    R[:2,:] = -R[:2,:]
    R = np.transpose(R)
    T[:2] = -T[:2]
    return R, T

def main():
    # Argument of conversion
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--img_w", default=512, type=int)
    parser.add_argument("--img_h", default=512, type=int)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    W, H = int(args.img_w), int(args.img_h)

    ROOT_DIR = args.dataset_dir
    scenes = sorted(os.listdir(ROOT_DIR))

    for scene in scenes:

        SCENE_DIR = os.path.join(ROOT_DIR, scene)
        POSE_DIR = os.path.join(SCENE_DIR, "pose")
        IMAGE_DIR = os.path.join(SCENE_DIR, "rgb")

        pose_file_list = sorted(os.listdir(POSE_DIR))
        image_file_list = sorted(os.listdir(IMAGE_DIR))
        # make sure the correspondence of pose and image is correct
        assert(len(pose_file_list) == len(image_file_list))
        for idx in range(len(pose_file_list)):
            pose_file_name = pose_file_list[idx].split('.')[0]
            image_file_name = image_file_list[idx].split('.')[0]
            assert(pose_file_name == image_file_name)

        OUTPUT_SCENE_DIR = os.path.join(OUTPUT_DIR, scene)
        OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_SCENE_DIR, 'train')
        OUTPUT_VALID_DIR = os.path.join(OUTPUT_SCENE_DIR, 'valid')
        os.makedirs(OUTPUT_SCENE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
        os.makedirs(OUTPUT_VALID_DIR, exist_ok=True)

        train_rot_mat_list = []
        train_trans_mat_list = []
        valid_rot_mat_list = []
        valid_trans_mat_list = []

        # convert camera pose (extrinsic)
        for idx, pose_file in enumerate(pose_file_list):

            transform_matrix = np.loadtxt(os.path.join(POSE_DIR, pose_file))
            R, T = convert_pose_nsvf_to_pytorch3d(transform_matrix)

            split = int(pose_file_name.split('_')[0])   # train: 0, valid: 1 for Tanks & Temples, 2 for Synthetic NeRF
            if split == 0:
                train_rot_mat_list.append(R)
                train_trans_mat_list.append(T)
            elif split == 1:
                valid_rot_mat_list.append(R)
                valid_trans_mat_list.append(T)

        train_rot_mat_list = np.array(train_rot_mat_list)
        train_trans_mat_list = np.array(train_trans_mat_list)
        valid_rot_mat_list = np.array(valid_rot_mat_list)
        valid_trans_mat_list = np.array(valid_trans_mat_list)

        np.save(os.path.join(OUTPUT_TRAIN_DIR, 'R.npy'), train_rot_mat_list)
        np.save(os.path.join(OUTPUT_TRAIN_DIR, 'T.npy'), train_trans_mat_list)
        np.save(os.path.join(OUTPUT_VALID_DIR, 'R.npy'), valid_rot_mat_list)
        np.save(os.path.join(OUTPUT_VALID_DIR, 'T.npy'), valid_trans_mat_list)

        # convert camera pose (intrinsic)
        cameras_txt_path = os.path.join(OUTPUT_SCENE_DIR, 'cameras.txt')
        intrinsic_path = os.path.join(SCENE_DIR, "intrinsics.txt")
        with open(cameras_txt_path, 'w') as f_out:
            # note that we store it by following COLMAP 'cameras.txt' format
            f_out.writelines(['# Camera list with one line of data per camera:\n', \
                            '# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n', \
                            '# Number of cameras: 1\n'])
            intrinsic_matrix = load_intrinsics(intrinsic_path)
            fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
            f_out.write('1 PINHOLE %d %d %f %f %f %f\n' % (W, H, fx, fy, cx, cy))

        # copy images to output folder
        os.makedirs(os.path.join(OUTPUT_TRAIN_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_VALID_DIR, 'images'), exist_ok=True)
        for image_file in image_file_list:
            file_name = image_file.split('.')[0]
            split = int(file_name.split('_')[0])    # train: 0, valid: 1 for Tanks & Temples, 2 for Synthetic NeRF
            if split == 0:
                target_image_path = os.path.join(OUTPUT_TRAIN_DIR, 'images', image_file)
            elif split == 1:
                target_image_path = os.path.join(OUTPUT_VALID_DIR, 'images', image_file)
            source_image_path = os.path.join(IMAGE_DIR, image_file)
            os.system('cp %s %s' % (source_image_path, target_image_path))


if __name__ == "__main__":
    main()