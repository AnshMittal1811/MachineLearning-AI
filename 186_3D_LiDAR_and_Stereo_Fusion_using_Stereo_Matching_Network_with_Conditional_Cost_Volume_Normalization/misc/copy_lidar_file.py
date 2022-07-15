
import os
import shutil

# Move KITTI Completion (sparse depth) to KITTI Stereo
# 200 images from 000000_10 (11) to 000199_10 (11) maps to raw data using devkit/mapping/train_mapping.txt
# Get devkit before running the script
#
# Directory Structure
#   copy_lidar_file.py
#   {src_dir}/{train,val}/{video_id}/proj_depth/velodyne_raw/image_02,image_03/{frame_index}.png
#   {tgt_dir}/data_scene_flow/training/{velodyne_2,velodyne_3}/{tgt_index.zfill(6)}.png
#   {devkit_dir}/mapping/train_mapping.txt
#
# Example:
#   "2011_09_26 2011_09_26_drive_0009_sync 0000000384" at Line 11 means:
#       The 10th image pair 000010_10.png (11) in training data dir 
#       maps to '2011_09_26_drive_0009_sync/0000000384.png' in raw data dir
#
# Program Behavior:
#   Copy file from
#       /home/johnson/Desktop/dataset/KITTI/data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/velodyne_raw/image_02/0000000384.png
#   Via
#       2011_09_26 2011_09_26_drive_0009_sync 0000000384 at Line 11
#   To
#       /home/johnson/Desktop/dataset/kitti_stereo/data_scene_flow/training/velodyne_2/000010.png
# 


def copy_file(src_path, tgt_path):
    if not os.path.isdir(os.path.dirname(tgt_path)):
        os.makedirs(os.path.dirname(tgt_path))
    shutil.copyfile(src_path, tgt_path)
    print('\tCopy {} to {}'.format(src_path, tgt_path))

def get_mapping_list(devkit_dir):
    with open(os.path.join(devkit_dir, 'mapping', 'train_mapping.txt'), 'r') as infile:
        mapping_lines = infile.readlines()
    return [mapl.split() for mapl in mapping_lines]

def cp_data(src_path_list, src_dir='KITTI/data_depth_annotated/', tgt_dir='kitti_stereo'):
    """ Read path from KITTI Depth Completion dataset """
    for tgt_index, src_path in enumerate(src_path_list):
        if len(src_path) != 3:
            # [] vs ['2011_10_03', '2011_10_03_drive_0047_sync', '0000000556']
            continue

        # Get path and file name
        date = src_path[0]
        video_id = src_path[1]
        frame_index = src_path[2]

        # Directory of ground truth depth maps
        velo_dir = os.path.join(src_dir, 'train', video_id, 'proj_depth', 'velodyne_raw')
        velo_file = '{}.png'.format(frame_index)
        depth_left_dir = os.path.join(velo_dir, 'image_02', velo_file)
        depth_right_dir = os.path.join(velo_dir, 'image_03', velo_file)

        # If file not found in the training set, use val set instead
        if not os.path.isfile(depth_left_dir):
            #print("Path not found {}".format(depth_left_dir))
            depth_left_dir = depth_left_dir.replace('train', 'val')
            depth_right_dir = depth_right_dir.replace('train', 'val')

        # Output path
        out_dir = os.path.join(tgt_dir, 'data_scene_flow', 'training')
        out_file = '{}.png'.format(str(tgt_index).zfill(6))
        out_left_dir = os.path.join(out_dir, 'velodyne_2', out_file)
        out_right_dir = os.path.join(out_dir, 'velodyne_3', out_file)

        # Copy files
        copy_file(depth_left_dir, out_left_dir)
        copy_file(depth_right_dir, out_right_dir)


if __name__ == '__main__':
    map_list = get_mapping_list('devkit')
    cp_data(map_list)

