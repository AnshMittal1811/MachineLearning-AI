import os

# preprocess = ['normalize', 'scale', 'rotate', 'jitter', 'kd_tree_idx']

classes = ["aero", "bag", "cap", "car", "chair",
           "earph", "guitar", "knife", "lamp", "laptop", "motor", "mug",
           "pistol", "rocket", "skate", "table"]
datset_dir = '/scratch/rahulsajnani/research/data/shapenet_single_class/'
# datset_dir = 'E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'val_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = os.path.join(datset_dir, 'data_hdf5')
val_data_folder = os.path.join(datset_dir, 'data_hdf5')
test_data_folder = os.path.join(datset_dir, 'data_hdf5')

ShapnetSingle_class = {'name': 'shapenet_single_class',
                          'num_classes': 1,
                          'classes': classes,
                          'train_data_folder': train_data_folder,
                          'val_data_folder': val_data_folder,
                          'test_data_folder': test_data_folder,
                          'train_files_list': train_files_list,
                          'val_files_list': val_files_list,
                          'test_files_list': test_files_list,
                          'train_preprocessing': [],
                          # 'train_preprocessing': [],
                          'val_preprocessing': [],
                          'test_preprocessing': []}

classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
           'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
           'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
           'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
           'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


def get_dataset_files(dataset_path):
   
    classes = ["aero", "bag", "cap", "car", "chair",
           "earph", "guitar", "knife", "lamp", "laptop", "motor", "mug",
           "pistol", "rocket", "skate", "table"]
           
    train_files_list = os.path.join(dataset_path, 'train_files.txt')
    val_files_list = os.path.join(dataset_path, 'val_files.txt')
    test_files_list = os.path.join(dataset_path, 'test_files.txt')

    train_data_folder = os.path.join(dataset_path)
    val_data_folder = os.path.join(dataset_path)
    test_data_folder = os.path.join(dataset_path)

    ShapenetSingle_class = {'name': 'shapenet_single_class',
                            'num_classes': 1,
                            'classes': classes,
                            'train_data_folder': train_data_folder,
                            'val_data_folder': val_data_folder,
                            'test_data_folder': test_data_folder,
                            'train_files_list': train_files_list,
                            'val_files_list': val_files_list,
                            'test_files_list': test_files_list,
                            'train_preprocessing': [],
                            # 'train_preprocessing': [],
                            'val_preprocessing': [],
                            'test_preprocessing': []}

    return ShapenetSingle_class

datset_dir = 'E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes'
# datset_dir = 'E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = os.path.join(datset_dir, 'data_hdf5')
val_data_folder = os.path.join(datset_dir, 'data_hdf5')
test_data_folder = os.path.join(datset_dir, 'data_hdf5')


"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original_rotated/data_hdf5')
"""

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
"""

modelnet40single_class = {'name': 'modelnet40single_class',
                               'num_classes': 1,
                               'classes': classes,
                               'train_data_folder': train_data_folder,
                               'val_data_folder': val_data_folder,
                               'test_data_folder': test_data_folder,
                               'train_files_list': train_files_list,
                               'val_files_list': val_files_list,
                               'test_files_list': test_files_list,
                               'train_preprocessing': ['rotate', 'kd_tree_idx'],
                               # 'train_preprocessing': [],
                               'val_preprocessing': ['rotate', 'kd_tree_idx'],
                               'test_preprocessing': ['rotate', 'kd_tree_idx']}

datset_dir = 'E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')



train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_normals/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_normals/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_normals/data_hdf5')



"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
"""

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
"""

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_fps_multiscale/data_hdf5')
"""

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
                               'val_preprocessing': ['rotate', 'kd_tree_idx'],
                               'test_preprocessing': ['rotate', 'kd_tree_idx']}

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

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_2048_original/data_hdf5')
"""

modelnet40rotate_z = {'name': 'modelnet40aligned',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['rotate_z', 'scale'],
                     'val_preprocessing': ['rotate_z'],
                     'test_preprocessing': ['rotate_z']}

modelnet40aligned = {'name': 'modelnet40canonical',
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

modelnet40aligned_test_rot = {'name': 'modelnet40aligned_test_rot',
                     'num_classes': 40,
                     'classes': classes,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['rotate_z', 'scale'],
                     'val_preprocessing': ['rotate'],
                     'test_preprocessing': ['rotate']}


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


datset_dir = 'E:/Users/Adrien/Documents/Datasets/ScanObjectNN_h5_files/main_split_nobg'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = datset_dir
val_data_folder = datset_dir
test_data_folder = datset_dir

"""
train_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
val_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
test_data_folder = os.path.join(datset_dir, 'modelnet40_hdf5_1024_original/data_hdf5')
"""

ScanObjectNN = {'name': 'ScanObjectNN',
                'num_classes': 15,
                'classes': classes,
                'train_data_folder': train_data_folder,
                'val_data_folder': val_data_folder,
                'test_data_folder': test_data_folder,
                'train_files_list': train_files_list,
                'val_files_list': val_files_list,
                'test_files_list': test_files_list,
                'train_preprocessing': ['rotate_z', 'scale'],
                'val_preprocessing': [],
                'test_preprocessing': []}


ScanObjectNNRot = {'name': 'ScanObjectNNRot',
                'num_classes': 15,
                'classes': classes,
                'train_data_folder': train_data_folder,
                'val_data_folder': val_data_folder,
                'test_data_folder': test_data_folder,
                'train_files_list': train_files_list,
                'val_files_list': val_files_list,
                'test_files_list': test_files_list,
                'train_preprocessing': ['rotate', 'scale'],
                'val_preprocessing': [],
                'test_preprocessing': []}


datset_dir = 'E:/Users/Adrien/Documents/Datasets/shapenet_classifiaction'

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = datset_dir
val_data_folder = datset_dir
test_data_folder = datset_dir

ShapenetClassifAligned = {'name': 'ShapeNetClassifAligned',
                'num_classes': 16,
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

# datsets_list = [modelnet40aligned_test_rot]

# datsets_list = [modelnet40rotated_augmented, modelnet40aligned]

# datsets_list = [modelnet40rotated_augmented]

# datsets_list = [modelnet40aligned_test_rot]

# datsets_list = [modelnet40aligned]

# datsets_list = [modelnet40single_class]

datsets_list = [ShapnetSingle_class]

# datsets_list = [ShapenetClassifAligned]




