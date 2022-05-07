import os

# preprocess = ['normalize', 'scale', 'rotate', 'jitter', 'kd_tree_idx']


classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
           'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
           'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
           'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
           'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

datset_dir = 'E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
"""

train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')

modelnet40rotated_augmented = {'name': 'modelnet40rotated_augmented',
                               'num_classes': 40,
                               'classes': classes,
                               'train_data_folder': train_data_folder,
                               'val_data_folder': val_data_folder,
                               'test_data_folder': test_data_folder,
                               'train_files_list': train_files_list,
                               'val_files_list': val_files_list,
                               'test_files_list': test_files_list,
                               # 'train_preprocessing': ['scale', 'rotate', 'kd_tree_idx'],
                               'train_preprocessing': ['rotate', 'scale', 'kd_tree_idx'],
                               'val_preprocessing': ['kd_tree_idx'],
                               'test_preprocessing': ['kd_tree_idx']}

modelnet40rotated = {'name': 'modelnet40aligned',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['scale', 'kd_tree_idx'],
                     'val_preprocessing': ['kd_tree_idx'],
                     'test_preprocessing': ['kd_tree_idx']}

train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')

modelnet40aligned = {'name': 'modelnet40aligned',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                      'train_preprocessing': ['scale', 'kd_tree_idx'],
                     'val_preprocessing': ['kd_tree_idx'],
                     'test_preprocessing': ['kd_tree_idx']}


                     # 'train_preprocessing': ['scale', 'kd_tree_idx'],
                     # 'val_preprocessing': ['kd_tree_idx'],
                     # 'test_preprocessing': ['kd_tree_idx']}


modelnet40aligned_test_rot = {'name': 'modelnet40aligned_test_rot',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['scale', 'kd_tree_idx'],
                     'val_preprocessing': ['rotate', 'kd_tree_idx'],
                     'test_preprocessing': ['rotate', 'kd_tree_idx']}


modelnet40rot_y = {'name': 'modelnet40rot_y',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['scale', 'rotate_y', 'kd_tree_idx'],
                     'val_preprocessing': ['kd_tree_idx'],
                     'test_preprocessing': ['kd_tree_idx']}


# datsets_list = [modelnet40rotated_augmented]

# datsets_list = [modelnet40rotated_augmented, modelnet40aligned]

datsets_list = [modelnet40rotated_augmented]

# datsets_list = [modelnet40aligned_test_rot]

"""
datset_dir = 'C:/Users/adrien/Documents/Datasets/ModelNet40_hdf5'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5_multires')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5_multires')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5_multires')

modelnet40rotated_multires = {'name': 'modelnet40aligned_multires',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': [],
                     'val_preprocessing': [],
                     'test_preprocessing': []}

datsets_list = [modelnet40rotated_multires]
"""