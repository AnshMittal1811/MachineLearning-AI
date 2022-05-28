import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import cupy as cp
import warnings
import chainer
from PIL import Image
import math
warnings.filterwarnings("ignore")
sys.path.append("./neural_renderer/")
import matplotlib.pyplot as plt
import nmr_test as nmr
import neural_renderer
from torchvision import transforms
from torchvision.transforms import functional as F

class MyDatasetTestAdv(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']
                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures_adv = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = faces[None, :, :]
        self.vertices_var = vertices[None, :, :]
        self.mask_renderer = nmr.NeuralRenderer(img_size=img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask

    def set_textures(self, textures_adv):
        self.textures_adv = textures_adv

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file, allow_pickle=True)  #.item()
        img = data['img']
        veh_trans, cam_trans = data['veh_trans'], data['cam_trans']

        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up

        imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures_adv)

        img = img[:, :, ::-1] 
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)

        imgs_pred = imgs_pred / torch.max(imgs_pred)

        if self.ret_mask:
            mask_file = os.path.join(self.mask_dir, "%s.png" % self.files[index][:-4])
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
            mask = torch.from_numpy(mask.astype('float32')).cuda()
            total_img = (1 - mask) * img + (255 * imgs_pred) * mask
            return index, total_img.squeeze(0), imgs_pred.squeeze(0), mask, self.files[index]
        total_img = img + 255 * imgs_pred
        return index, total_img.squeeze(0), imgs_pred.squeeze(0), self.files[index]

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    rnder = neural_renderer.Renderer()
    vertices = np.expand_dims(vertices, axis=0)
    faces = np.expand_dims(faces, axis=0)
    textures = np.expand_dims(textures, axis=0)
    faces = chainer.Variable(chainer.cuda.to_gpu(faces, 0))
    vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, 0))
    textures = chainer.Variable(chainer.cuda.to_gpu(textures, 0))
    image = rnder.render(vertices, faces, textures)
    image = image.data[0]
    image = (np.clip(cp.asnumpy(image),0,1) * 255).astype(np.uint8)
    image = Image.fromarray(np.transpose(image, (1,2,0)))
    image.show()
    dataset = MyDataset('../data/phy_attack/train/', 608, 4, faces, vertices)
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    
    for img, car_box in loader:
        print(img.size(), car_box.size())
