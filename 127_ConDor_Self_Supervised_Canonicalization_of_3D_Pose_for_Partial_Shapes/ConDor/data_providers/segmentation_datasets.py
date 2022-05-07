import os

datset_dir = 'C:/Users/adrien/Documents/Datasets/SIG2017_toricconv_seg_hdf5'

parts = ['head', 'hands', 'forearms', 'arms', 'body', 'legs', 'forelegs', 'feets']

train_files_list = os.path.join(datset_dir, 'train_files.txt')
val_files_list = os.path.join(datset_dir, 'test_files.txt')
test_files_list = os.path.join(datset_dir, 'test_files.txt')

train_data_folder = os.path.join(datset_dir, 'data')
val_data_folder = os.path.join(datset_dir, 'data')
test_data_folder = os.path.join(datset_dir, 'data')

sig17_human_seg_augmented = {'name': 'sig17_human_seg_augmented',
                               'num_parts': 8,
                               'parts': [],
                               'train_data_folder': train_data_folder,
                               'val_data_folder': val_data_folder,
                               'test_data_folder': test_data_folder,
                               'train_files_list': train_files_list,
                               'val_files_list': val_files_list,
                               'test_files_list': test_files_list,
                               'train_preprocessing': ['scale', 'kd_tree_idx'],
                               'val_preprocessing': ['kd_tree_idx'],
                               'test_preprocessing': ['rotate', 'kd_tree_idx']}

# datasets_list = [sig17_human_seg_augmented]





dataset_dir = 'E:/Users/Adrien/Documents/Datasets/shapenet_segmentation/hdf5_data'

train_files_list = os.path.join(dataset_dir, 'train_hdf5_file_list.txt')
val_files_list = os.path.join(dataset_dir, 'val_hdf5_file_list.txt')
test_files_list = os.path.join(dataset_dir, 'test_hdf5_file_list.txt')

"""
train_data_folder = os.path.join(datset_dir, 'data')
val_data_folder = os.path.join(datset_dir, 'data')
test_data_folder = os.path.join(datset_dir, 'data')
"""

train_data_folder = dataset_dir
val_data_folder = dataset_dir
test_data_folder = dataset_dir

# cat_to_labels = os.path.join(dataset_dir, 'catid_partid_to_overallid.json')
labels_to_cat = os.path.join(dataset_dir, 'overallid_to_catid_partid.json')

shapenet_seg_augmented = {'name': 'shapenet_seg_augmented',
                          'num_parts': 50,
                          'num_classes': 16,
                          'parts': [],
                          # 'cat_to_labels': cat_to_labels,
                          'labels_to_cat': labels_to_cat,
                          'train_data_folder': train_data_folder,
                          'val_data_folder': val_data_folder,
                          'test_data_folder': test_data_folder,
                          'train_files_list': train_files_list,
                          'val_files_list': val_files_list,
                          'test_files_list': test_files_list,
                          'train_preprocessing': ['rotate', 'scale'],
                          'val_preprocessing': ['rotate'],
                          'test_preprocessing': ['rotate']}

shapenet_seg = {'name': 'shapenet_seg',
                'num_parts': 50,
                'num_classes': 16,
                'parts': [],
                # 'cat_to_labels': cat_to_labels,
                'labels_to_cat': labels_to_cat,
                'train_data_folder': train_data_folder,
                'val_data_folder': val_data_folder,
                'test_data_folder': test_data_folder,
                'train_files_list': train_files_list,
                'val_files_list': val_files_list,
                'test_files_list': test_files_list,
                'train_preprocessing': ['rotate_z', 'scale'],
                'val_preprocessing': ['rotate_z'],
                'test_preprocessing': ['rotate_z']}

dataset_dir = 'E:/Users/Adrien/Documents/Datasets/shapenet_single_class'


train_files_list = os.path.join(dataset_dir, 'train_files.txt')
val_files_list = os.path.join(dataset_dir, 'val_files.txt')
test_files_list = os.path.join(dataset_dir, 'test_files.txt')


train_data_folder = os.path.join(dataset_dir, 'data_hdf5')
val_data_folder = os.path.join(dataset_dir, 'data_hdf5')
test_data_folder = os.path.join(dataset_dir, 'data_hdf5')



shapenet_singleclass_seg = {'name': 'shapenet_seg_singleclass',
                'num_parts': 2,
                'num_classes': 1,
                'parts': [],
                # 'cat_to_labels': cat_to_labels,
                'labels_to_cat': labels_to_cat,
                'train_data_folder': train_data_folder,
                'val_data_folder': val_data_folder,
                'test_data_folder': test_data_folder,
                'train_files_list': train_files_list,
                'val_files_list': val_files_list,
                'test_files_list': test_files_list,
                'train_preprocessing': ['rotate', 'scale'],
                'val_preprocessing': ['rotate'],
                'test_preprocessing': ['rotate']}


dataset_dir = 'C:/Users/adrien/Documents/Datasets/RF00001-family/hdf5_data'

train_files_list = os.path.join(dataset_dir, 'train_hdf5_file_list.txt')
val_files_list = os.path.join(dataset_dir, 'val_hdf5_file_list.txt')
test_files_list = os.path.join(dataset_dir, 'test_hdf5_file_list.txt')

"""
train_data_folder = os.path.join(datset_dir, 'data')
val_data_folder = os.path.join(datset_dir, 'data')
test_data_folder = os.path.join(datset_dir, 'data')
"""

train_data_folder = dataset_dir
val_data_folder = dataset_dir
test_data_folder = dataset_dir

# cat_to_labels = os.path.join(dataset_dir, 'catid_partid_to_overallid.json')
# labels_to_cat = os.path.join(dataset_dir, 'overallid_to_catid_partid.json')

rna_seg_augmented = {'name': 'rna_seg_augmented',
                     'num_parts': 160,
                     'num_classes': 1,
                     'parts': [],
                     # 'cat_to_labels': cat_to_labels,
                     'labels_to_cat': None,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['rotate', 'scale', 'kd_tree_idx'],
                     'val_preprocessing': ['kd_tree_idx'],
                     'test_preprocessing': ['kd_tree_idx']}

rna_seg = {'name': 'rna_seg',
           'num_parts': 160,
           'num_classes': 1,
           'parts': [],
           # 'cat_to_labels': cat_to_labels,
           'labels_to_cat': None,
           'train_data_folder': train_data_folder,
           'val_data_folder': val_data_folder,
           'test_data_folder': test_data_folder,
           'train_files_list': train_files_list,
           'val_files_list': val_files_list,
           'test_files_list': test_files_list,
           'train_preprocessing': ['scale', 'kd_tree_idx'],
           'val_preprocessing': ['kd_tree_idx'],
           'test_preprocessing': ['kd_tree_idx']}


# datasets_list = [rna_seg, rna_seg_augmented]

datasets_list = [shapenet_singleclass_seg]


