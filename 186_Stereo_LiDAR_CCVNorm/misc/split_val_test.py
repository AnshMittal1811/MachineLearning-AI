import os
import random
import shutil

def get_datapath(depth_dir, mode='val'):
    """ Read path to all data from KITTI Depth Completion dataset """
    left_data_path = {'sdepth': [], 'depth': []}
    right_data_path = {'sdepth': [], 'depth': []}
    dir_name_list = sorted(os.listdir(os.path.join(depth_dir, mode)))
    for dir_name in dir_name_list:
        # Directory of ground truth depth maps
        depth_left_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'groundtruth', 'image_02')
        depth_right_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'groundtruth', 'image_03')
        # Directory of sparse depth maps from LiDAR
        sdepth_left_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'velodyne_raw', 'image_02')
        sdepth_right_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'velodyne_raw', 'image_03')

        # Get image names (DO NOT obtain from raw data directory since the annotated data is pruned)
        file_name_list = sorted(os.listdir(depth_left_dir))

        for file_name in file_name_list:
            # Path to ground truth depth maps
            depth_left_path = os.path.join(depth_left_dir, file_name)
            depth_right_path = os.path.join(depth_right_dir, file_name)
            # Path to sparse depth maps
            sdepth_left_path = os.path.join(sdepth_left_dir, file_name)
            sdepth_right_path = os.path.join(sdepth_right_dir, file_name)

            # Add to list
            left_data_path['sdepth'].append(sdepth_left_path)
            left_data_path['depth'].append(depth_left_path)
            right_data_path['sdepth'].append(sdepth_right_path)
            right_data_path['depth'].append(depth_right_path)

    return left_data_path, right_data_path

def copy_file(data_list, idx, new_dirname, verbose=True):
    src_path = data_list[i]
    tgt_path = data_list[i].split('/')
    tgt_path[2] = new_dirname
    tgt_dir = os.path.join(*tgt_path[:-1])
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    tgt_path = os.path.join(*tgt_path)
    shutil.copyfile(src_path, tgt_path)
    print('\tCopy {} to {}'.format(src_path, tgt_path))

left_data_path, right_data_path = get_datapath('../data_depth_annotated/', 'test_source')
indices = list(range(len(left_data_path['sdepth'])))
random.seed(100)
random.shuffle(indices)
val_indices = indices[:1000]
my_val_name = 'my_test'
for ii, i in enumerate(val_indices): # For my_val
    print('[my_val][{}/{}]'.format(ii, len(val_indices)-1))
    copy_file(left_data_path['sdepth'], i, my_val_name)
    copy_file(left_data_path['depth'], i, my_val_name)
    copy_file(right_data_path['sdepth'], i, my_val_name)
    copy_file(right_data_path['depth'], i, my_val_name)
'''
test_indices = indices[1000:2000]
for ii, i in enumerate(test_indices): # For my_test
    print('[my_test][{}/{}]'.format(ii, len(test_indices)-1))
    copy_file(left_data_path['sdepth'], i, 'my_test')
    copy_file(left_data_path['depth'], i, 'my_test')
    copy_file(right_data_path['sdepth'], i, 'my_test')
    copy_file(right_data_path['depth'], i, 'my_test')
'''
