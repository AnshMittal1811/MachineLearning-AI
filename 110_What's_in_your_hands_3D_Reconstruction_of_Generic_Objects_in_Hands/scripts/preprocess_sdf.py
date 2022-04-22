# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import sys
import os
import os.path as osp
import json
import pickle
from tqdm import tqdm
from glob import glob


def water_tight(inp_file, out_file, reso=5000):
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    if not osp.exists(out_file):
        cmd = '/private/home/yufeiy2/Tools/Manifold/build/manifold %s %s %d' % (inp_file, out_file, reso)
        print(cmd)
        os.system(cmd)
    return out_file


def cp_folder(file_list, index_list, dst_dir, watertight=False):
    dst_list = []
    for mesh_file, index in zip(file_list, index_list):
        dst_file = osp.join(dst_dir, index + '.obj') 
        os.makedirs(osp.dirname(dst_file), exist_ok=True)
        if watertight:
            water_tight(mesh_file, dst_file)
        else:
            cmd = 'cp %s %s' % (mesh_file, dst_file)
            os.system(cmd)
            print(cmd)
        dst_list.append(dst_file)
    return dst_list


def generate_split(src_name, sdf_dir, data_dir):
    """for sdf preprocess"""
    if src_name == 'obman':
        get_index_list = make_mesh_obman
    elif src_name == 'ho3d':
        get_index_list = make_mesh_ho3d
    elif src_name == 'mow':
        get_index_list = make_mesh_mow
    else:
        raise NotImplementedError(src_name)

    index_list = get_index_list(osp.join(sdf_dir, src_name, 'all'), data_dir)

    source_name = '%s_%s' % (src_name, 'all')
    index_dict = {src_name: {'all': index_list}}
    split_file = os.path.join(sdf_dir, '%s.json' % source_name)
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, 'w') as fp:
        json.dump(index_dict, fp, indent=4)


def make_mesh_mow(sdf_dir, data_dir):
    mesh_file = osp.join(data_dir, 'mow/results', '*', '*_norm.obj')
    source_name_list = glob(mesh_file)
    index_list = [e.split('/')[-2] for e in source_name_list]
    cp_folder(source_name_list, index_list, sdf_dir)
    return index_list


def make_mesh_ho3d(sdf_dir, data_dir):
    mesh_file = osp.join(data_dir, 'ho3dobj/models', '*', 'textured_simple.obj')
    source_name_list = glob(mesh_file)
    index_list = [e.split('/')[-2] for e in source_name_list]
    cp_folder(source_name_list, index_list, sdf_dir)
    return index_list


def make_mesh_obman(sdf_dir='data/sdf/MeshInp/obman/all', data_dir='data/'):
    def get_cad_index(data_dir, split):
        index_list = [line.strip() for line in open(osp.join(data_dir, '%s.txt' % split))]
        meta_dir = os.path.join(data_dir, split, 'meta', '{}.pkl')
        cad_list= []
        for index in tqdm(index_list):
            meta_path = meta_dir.format(index)
            with open(meta_path, "rb") as meta_f:
                meta_info = pickle.load(meta_f)

            cad_list.append(osp.join(meta_info["class_id"], meta_info["sample_id"]))
        return cad_list
    src_dir = data_dir + '/obmanobj/' 
    shape_dir = os.path.join(src_dir, '{}', 'models', 'model_normalized.obj')
    index_list = []

    for split in ['test', 'val', 'train']:
        index_list_each = get_cad_index(data_dir+ '/obman', split)
        index_list += list(set(index_list_each))
    
    src_list = [shape_dir.format(e) for e in index_list]
    cp_folder(src_list, index_list, sdf_dir, watertight=True)

    return index_list



if __name__ == '__main__':
    src = sys.argv[1]
    data_dir = sys.argv[2]
    sdf_dir = sys.argv[3] + '/MeshInp'
    print(src)
    generate_split( src, sdf_dir, data_dir)

