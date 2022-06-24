import numpy as np
import scipy.io as sio
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
from src.dataset.augmentation_old import att_aug2
import cv2
import os
import config
import math
import pickle
import os
from config import *
from src.dataset.uv_face import mean_shape_map_np, face_mask_np, foreface_ind
from io import BytesIO


class ImageData:
    def __init__(self):
        self.image_path = ""
        self.info_path = ""
        self.uv_posmap_path = ""
        self.attention_path = ""

        self.t_shape2face = None
        self.t_face2shape = None
        self.pos_map = None
        self.attention_mask = None
        self.image = None

    def read_path(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.image_path = image_dir + '/' + image_name + '.jpg'
        self.uv_posmap_path = image_dir + '/' + image_name + '_pos_map.npy'
        self.info_path = image_dir + '/' + image_name + '_info.mat'
        self.attention_path = image_dir + '/' + image_name + '_attention.jpg'

        info = sio.loadmat(self.info_path)
        self.t_shape2face = info['t_shape2face']
        self.t_face2shape = info['t_face2shape']

    def get_image(self):
        if self.image is None:
            return np.array(Image.open(self.image_path).convert('RGB'))
        else:
            return np.array(Image.open(BytesIO(self.image)).convert('RGB'))

    def get_pos_map(self):
        if self.pos_map is None:
            pos_map = np.load(self.uv_posmap_path).astype(np.float32)
        else:
            pos_map = self.pos_map.astype(np.float32)

        # mean_z = pos_map[foreface_ind[:, 0], foreface_ind[:, 1], 2].mean()
        # pos_map[:, :, 2] -= mean_z
        # pos_map[:, :, 2] += CROPPED_IMAGE_SIZE / 2

        return pos_map

    def get_attention_mask(self):
        if self.attention_mask is None:
            return np.array(Image.open(self.attention_path))
        else:
            return np.array(Image.open(BytesIO(self.attention_mask)))

    def get_transmat_face2shape(self):
        return self.t_face2shape

    def get_transmat_shape2face(self):
        return self.t_shape2face


class FaceRawDataset:
    def __init__(self):
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def add_image_data(self, data_dir, add_mode='train', split_rate=0.8, save_posmap_num=0):
        all_data = []
        saved_num = 0
        if os.path.exists(f'{data_dir}/all_image_data.pkl'):
            all_data = self.load_image_data_paths(data_dir)
        else:
            for root, dirs, files in os.walk(data_dir):
                dirs.sort()  # keep order in linux
                for dir_name in dirs:
                    image_name = dir_name
                    if not os.path.exists(root + '/' + dir_name + '/' + image_name + '.jpg'):
                        print('skip ', root + '/' + dir_name)
                        continue
                    temp_image_data = ImageData()
                    temp_image_data.read_path(root + '/' + dir_name)
                    if saved_num < save_posmap_num:
                        saved_num += 1
                        temp_image_data.pos_map = np.load(temp_image_data.uv_posmap_path)

                    all_data.append(temp_image_data)
                    print(f'\r{len(all_data)}', end='')

        print(len(all_data), 'data added')

        if add_mode == 'train':
            self.train_data.extend(all_data)
        elif add_mode == 'val':
            self.val_data.extend(all_data)
        elif add_mode == 'both':
            num_train = math.floor(len(all_data) * split_rate)
            self.train_data.extend(all_data[0:num_train])
            self.val_data.extend(all_data[num_train:])
        elif add_mode == 'test':
            self.test_data.extend(all_data)
        if not os.path.exists(f'{data_dir}/all_image_data.pkl'):
            self.save_image_data_paths(all_data, data_dir)

    def save_image_data_paths(self, all_data, data_dir):
        print('saving data path list')
        ft = open(f'{data_dir}/all_image_data.pkl', 'wb')
        pickle.dump(all_data, ft)
        ft.close()
        print('data path list saved')

    def load_image_data_paths(self, data_dir):
        print('loading data path list')
        ft = open(f'{data_dir}/all_image_data.pkl', 'rb')
        all_data = pickle.load(ft)
        ft.close()
        print('data path list loaded')
        return all_data


def img_to_tensor(image):
    return torch.from_numpy(image.transpose((2, 0, 1)))


def uv_map_to_tensor(uv_map):
    return torch.from_numpy(uv_map.transpose((2, 0, 1)))


class FaceDataGenerator(Dataset):
    def __init__(self, all_image_data, generate_mode='IPOA', is_aug=False):
        super(FaceDataGenerator, self).__init__()
        self.all_image_data = all_image_data
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3
        # mode=posmap or offset
        self.mode = generate_mode
        self.is_aug = is_aug

        self.toTensor = transforms.ToTensor()
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    def __getitem__(self, index):
        if self.mode == 'IPOA':
            item = self.all_image_data[index]
            image = (item.get_image() / 255.0).astype(np.float32)
            pos_map = (item.get_pos_map())

            t_face2shape = item.get_transmat_face2shape()
            attention_mask = item.get_attention_mask() / 255.0

            uvm4d = np.concatenate((pos_map, np.ones((256, 256, 1))), axis=-1)
            shape_map = uvm4d.dot(t_face2shape.T)[:, :, :3] * OFFSET_FIX_RATE
            offset_map = shape_map - mean_shape_map_np

            # uvm4d = np.concatenate((uv_position_map, np.ones((256, 256, 1))), axis=-1)
            # t_all = (T_bfm.T.dot(T_3d.T)).T
            # shape = (uv_offset_map + mean_shape_map) / OFFSET_FIX_RATE
            # shape4d = np.concatenate((shape, np.ones((256, 256, 1))), axis=-1)
            # print(shape4d.dot(t_all.T) - uvm4d)
            # print(uvm4d.dot(inv(t_all).T) - shape4d)

            if self.is_aug:
                image, pos_map, attention_mask = att_aug2(image, pos_map, attention_mask)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = img_to_tensor(image)
            else:
                # image = augmentation.randomErase(image)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = img_to_tensor(image)

            pos_map = pos_map / POSMAP_FIX_RATE
            if abs(offset_map).max() > 1:
                print('\n too large offset', abs(offset_map).max())

            pos_map = uv_map_to_tensor(pos_map)
            offset_map = uv_map_to_tensor(offset_map)

            attention_mask = Image.fromarray(attention_mask)
            attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
            attention_mask = np.array(attention_mask)
            attention_mask = torch.from_numpy(attention_mask).unsqueeze(0)
            return image.float(), pos_map.float(), offset_map.float(), attention_mask.float()

        else:
            return None

    def __len__(self):
        return len(self.all_image_data)


def make_data_loader(all_image_data, mode='IPOA', batch_size=32, is_shuffle=False, is_aug=False, num_worker=8):
    dataset = FaceDataGenerator(all_image_data, mode, is_aug)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_worker,
                        drop_last=False, pin_memory=True)
    return loader


def make_dataset(folders, mode='train'):
    raw_dataset = FaceRawDataset()
    for folder in folders:
        raw_dataset.add_image_data(folder, mode)
    return raw_dataset
