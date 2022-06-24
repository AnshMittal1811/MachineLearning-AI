import os
import numpy as np
import scipy.io as sio
from skimage import io, transform
import skimage
from src.faceutil import mesh
import argparse
import ast
import copy
import multiprocessing
import math
from config import *

from src.dataset.uv_face import uv_coords
from src.dataset.base_face import bfm, get_transform_matrix
# from data import bfm2Mesh, mesh2UVmap, UVmap2Mesh, renderMesh
from numpy.linalg import inv
from src.util.mask import get_image_attention_mask, get_image_attention_mask_v2
from src.util.timer import TIMER


class DataProcessor:
    def __init__(self, bbox_extend_rate=1.5, marg_rate=0.1):

        print('bfm model loaded')

        self.image_file_name = ''
        self.image_name = ''
        self.image_path = ''
        self.image_dir = ''
        self.output_dir = ''  # output_dir/image_name/image_name_xxxx.xxx
        self.write_dir = ''  # write_dir/image_name_xxxx.xxx

        self.init_image = None
        self.image_shape = None
        self.bfm_info = None
        self.uv_position_map = None
        self.uv_texture_map = None
        self.mesh_info = None

        self.bbox_extend_rate = bbox_extend_rate
        self.marg_rate = marg_rate

    def initialize(self, image_path, output_dir='data/temp'):
        self.image_path = image_path
        self.image_file_name = image_path.strip().split('/')[-1]
        self.image_name = self.image_file_name.split('.')[0]
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            print('mkdir ', output_dir)
            os.mkdir(output_dir)
        if not os.path.exists(output_dir + '/' + self.image_name):
            os.mkdir(output_dir + '/' + self.image_name)
        self.write_dir = output_dir + '/' + self.image_name

        self.init_image = io.imread(self.image_path) / 255.
        self.image_shape = self.init_image.shape

    @staticmethod
    def get_bbox(kpt):
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        return left, top, right, bottom

    def get_crop_box(self, bbox):
        [left, top, right, bottom] = bbox
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        old_size = (right - left + bottom - top) / 2.0
        size = int(old_size * self.bbox_extend_rate)  # 1.5
        marg = old_size * self.marg_rate  # 0.1
        t_x = np.random.rand() * marg * 2 - marg
        t_y = np.random.rand() * marg * 2 - marg
        center[0] = center[0] + t_x
        center[1] = center[1] + t_y
        size = size * (np.random.rand() * 2 * self.marg_rate - self.marg_rate + 1)
        return center, size

    def run_offset_posmap(self):
        # TIMER.mark('a')
        # 1. load image and fitted parameters
        [height, _, _] = self.image_shape
        pose_para = self.bfm_info['Pose_Para'].T.astype(np.float32)
        shape_para = self.bfm_info['Shape_Para'].astype(np.float32)
        exp_para = self.bfm_info['Exp_Para'].astype(np.float32)
        vertices = bfm.generate_vertices(shape_para, exp_para)
        # offset_vertices = bfm.generate_offset(shape_para, exp_para)
        #  TIMER.mark('b')

        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]

        T_bfm = get_transform_matrix(s, angles, t, height)
        temp_ones_vec = np.ones((len(vertices), 1))
        homo_vertices = np.concatenate((vertices, temp_ones_vec), axis=-1)
        image_vertices = homo_vertices.dot(T_bfm.T)[:, 0:3]

        # 3. crop image with key points
        # 3.1 get old bbox
        kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
        [left, top, right, bottom] = self.get_bbox(kpt)
        old_bbox = np.array([[left, top], [right, bottom]])

        # 3.2 add margin to bbox
        [center, size] = self.get_crop_box([left, top, right, bottom])

        # 3.3 crop and record the transform parameters
        crop_h, crop_w = CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE

        T_3d = np.zeros((4, 4))
        T_3d[0, 0] = crop_w / size
        T_3d[1, 1] = crop_h / size
        T_3d[2, 2] = crop_w / size
        T_3d[3, 3] = 1.

        T_3d[0:3, 3] = [(size / 2 - center[0]) * crop_w / size, (size / 2 - center[1]) * crop_h / size,
                        -np.min(image_vertices[:, 2]) * crop_w / size]
        T_2d = np.zeros((3, 3))
        T_2d[0:2, 0:2] = T_3d[0:2, 0:2]
        T_2d[2, 2] = 1.
        T_2d[0:2, 2] = T_3d[0:2, 3]

        T_2d_inv = inv(T_2d)
        cropped_image = skimage.transform.warp(self.init_image, T_2d_inv, output_shape=(crop_h, crop_w))
        # 3.4 transform face position(image vertices)

        p4d = np.concatenate((image_vertices, temp_ones_vec), axis=-1)
        position = p4d.dot(T_3d.T)[:, 0:3]

        # offset_position = offset_vertices * OFFSET_FIX_RATE

        # 4. uv position map: render position in uv space
        uv_h, uv_w, uv_c = UV_MAP_SIZE, UV_MAP_SIZE, 3

        # TIMER.mark('c')

        uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h,
                                                    uv_w, uv_c)

        # uv_offset_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, offset_position, uv_h,
        #                                           uv_w, uv_c)

        # get new bbox
        kpt = position[bfm.kpt_ind, :].astype(np.int32)
        [left, top, right, bottom] = self.get_bbox(kpt)
        bbox = np.array([[left, top], [right, bottom]])

        # get gt landmark68
        init_kpt = image_vertices[bfm.kpt_ind, :]
        init_kpt_4d = np.concatenate((init_kpt, np.ones((68, 1))), axis=-1)
        new_kpt = init_kpt_4d.dot(T_3d.T)[:, 0:3]

        # 5. save files

        # TIMER.mark('d')
        attention_mask = get_image_attention_mask(uv_position_map)

        t_all = (T_bfm.T.dot(T_3d.T)).T
        t_all_inv = inv(t_all)
        # TIMER.mark('e')

        return attention_mask, uv_position_map, cropped_image, t_all, t_all_inv

    def process_item(self, image_path, output_dir):
        self.initialize(image_path, output_dir)
        self.bfm_info = sio.loadmat(self.image_path.replace('.jpg', '.mat'))

        result = self.run_offset_posmap()
        self.save_result(result)
        self.clear()

    def save_result(self, result):
        attention_mask, uv_position_map, cropped_image, t_all, t_all_inv = result
        io.imsave(self.write_dir + '/' + self.image_name + '_attention.jpg', attention_mask.astype(np.uint8))
        sio.savemat(self.write_dir + '/' + self.image_name + '_info.mat',
                    {'t_shape2face': t_all, 't_face2shape': t_all_inv})
        np.save(self.write_dir + '/' + self.image_name + '_pos_map.npy', uv_position_map.astype(np.float32))
        # np.save(self.write_dir + '/' + self.image_name + '_offset_map.npy', uv_offset_map.astype(np.float32))
        io.imsave(self.write_dir + '/' + self.image_name + '.jpg',
                  (np.squeeze(cropped_image * 255.0)).astype(np.uint8))

    def clear(self):
        self.image_file_name = ''
        self.image_name = ''
        self.image_path = ''
        self.image_dir = ''
        self.output_dir = ''
        self.write_dir = ''

        self.init_image = None
        self.image_shape = None
        self.bfm_info = None
        self.uv_position_map = None
        self.uv_texture_map = None
        self.mesh_info = None


def worker_process(image_paths, output_dirs, worker_id):
    print('worker:', worker_id, 'start. task number:', len(image_paths))
    data_processor = DataProcessor(bbox_extend_rate=BBOX_EXTEND_RATE, marg_rate=MARG_RATE)

    for i in range(len(image_paths)):
        # print('\r worker ' + str(id) + ' task ' + str(i) + '/' + str(len(image_paths)) +''+  image_paths[i])
        print("worker {} task {}/{}  {}\r".format(str(worker_id), str(i), str(len(image_paths)), image_paths[i]),
              end='')
        # output_list[id] = "worker {} task {}/{}  {}".format(str(id), str(i), str(len(image_paths)), image_paths[i])
        data_processor.process_item(image_paths[i], output_dirs[i])
    print('worker:', worker_id, 'end')


def multi_process(worker_num=WORKER_NUM, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    image_path_list = []
    output_dir_list = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(input_dir)
    for root, dirs, files in os.walk(input_dir):
        temp_output_dir = output_dir
        # tokens = root.split(input_dir)
        if not (root.split(input_dir)[1] == ''):
            temp_output_dir = output_dir + root.split(input_dir)[1]
            if not os.path.exists(temp_output_dir):
                os.mkdir(temp_output_dir)

        for file in files:
            file_tokens = file.split('.')
            file_type = file_tokens[1]
            if file_type == 'jpg' or file_type == 'png':
                image_path_list.append(root + '/' + file)
                output_dir_list.append(temp_output_dir)

    total_task = len(image_path_list)
    print('found images:', total_task)

    if worker_num <= 1:
        worker_process(image_path_list, output_dir_list, 0)
    elif worker_num > 1:
        jobs = []
        task_per_worker = math.ceil(total_task / worker_num)
        st_idx = [task_per_worker * i for i in range(worker_num)]
        ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
        for i in range(worker_num):
            # temp_data_processor = copy.deepcopy(data_processor)
            p = multiprocessing.Process(target=worker_process, args=(
                image_path_list[st_idx[i]:ed_idx[i]],
                output_dir_list[st_idx[i]:ed_idx[i]], i))
            jobs.append(p)
            p.start()


def run_mean_posmap():
    shape_mu = bfm.get_mean_shape() * OFFSET_FIX_RATE
    uv_mean_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, shape_mu, UV_MAP_SIZE,
                                            UV_MAP_SIZE, 3)
    np.save(UV_MEAN_SHAPE_PATH, uv_mean_map)


if __name__ == "__main__":
    run_mean_posmap()
    multi_process()
