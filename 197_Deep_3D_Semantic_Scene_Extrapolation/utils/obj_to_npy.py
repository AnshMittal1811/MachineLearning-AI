# create .npy voxelized file from each object.obj

# ----------------------------------------------------------------------------------

import glob
import os
import csv
import math

from binvox_parser import *

# ----------------------------------------------------------------------------------

model_category_mapping = []
models = []
voxel_size = 6  # cm
objects_voxel_size_dict = {}

# ----------------------------------------------------------------------------------

def csv_loader():
    with open('meta_data/ModelCategoryMapping.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            model_category_mapping.append(row)

    with open('meta_data/models.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            models.append(row)

# ----------------------------------------------------------------------------------

def compute_size(input_obj_file):
    file_name = str(input_obj_file)[0:-4]
    min_point = []
    max_point = []

    for model in models:
        if str(model["id"]) == file_name:
            min_point = model["minPoint"].split(",")
            max_point = model["maxPoint"].split(",")

    # map from str to int
    if min_point and max_point:
        min_point = map(float, min_point)
        max_point = map(float, max_point)

    # TODO: vectorized
    # find the longest dimension
    x_dim = math.sqrt((min_point[0] - max_point[0]) ** 2)
    y_dim = math.sqrt((min_point[1] - max_point[1]) ** 2)
    z_dim = math.sqrt((min_point[2] - max_point[2]) ** 2)
    long_dim = max(x_dim, y_dim, z_dim)
    size = int((long_dim * 100) / voxel_size)
    objects_voxel_size_dict[str(file_name)] = size
    return size

# ----------------------------------------------------------------------------------

def obj_to_binvox(input_obj_file):
    os.system("binvox.exe -d " + str(compute_size(input_obj_file)) + " " + str(input_obj_file))
    os.remove(input_obj_file)

# ----------------------------------------------------------------------------------

def binvox_to_npy(input_binvox_file):
    with open(input_binvox_file, 'rb') as f:
        voxel_model = read_as_coord_array(f)
    resolution = objects_voxel_size_dict[str(input_binvox_file[:-7])]
    voxel_model = sparse_to_dense(voxel_model.data, resolution)
    np.save(str(input_binvox_file[:-7]) + ".npy", voxel_model)
    os.remove(binvox_file)

# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # obj_to_binvox
    csv_loader()
    for obj_file in glob.glob('*.obj'):
        obj_to_binvox(obj_file)

    # binvox_to_npy
    for binvox_file in glob.glob('*.binvox'):
        binvox_to_npy(binvox_file)
