import os
import argparse
import vedo
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from .utils import random_sample_points, get_image_tensors

def get_camera_intrinsic(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[3].split(' ')
        W, H = int(lines[2]), int(lines[3])
        fx, fy = float(lines[4]), float(lines[5])
        px, py = float(lines[6]), float(lines[7])
    return [W, H, fx, fy, px, py]

def screen_to_ndc(intrinsic):
    W, H, fx, fy, px, py = intrinsic
    # convert from screen space to NDC space
    half_w, half_h = W/2, H/2
    fx_new = fx / half_w
    fy_new = fy / half_h
    px_new = -(px - half_w) / half_w
    py_new = -(py - half_h) / half_h
    return [W, H, fx_new, fy_new, px_new, py_new]
class TanksAndTemplesDataset(Dataset):
    def __init__(
        self,
        folder:str,
        read_points:bool=False,
        sample_rate:float=0.1,
        batch_points:int=10000
    ):
        R = np.load(os.path.join(folder, 'R.npy'))
        T = np.load(os.path.join(folder, 'T.npy'))
        R, T = torch.tensor(R), torch.tensor(T)
        self.R = R 
        self.T = T

        intrinsic = get_camera_intrinsic(os.path.join(folder, 'cameras.txt'))
        self.intrinsic = intrinsic

        # convert to NDC coordinates
        intrinsic = screen_to_ndc(intrinsic)
        W, H, fx, fy, px, py = intrinsic

        self.focal_length = ((fx,fy),)
        self.principal_point = ((px,py),)
        cameras = []
        for i in range(R.size(0)):
            cam = PerspectiveCameras(
                focal_length=((fx,fy),),
                principal_point=((px,py),),
                R=R[i][None],
                T=T[i][None],
                # image_size=((H,W),),
                # in_ndc=True
            )
            cameras.append(cam)
        self.cameras = cameras

        images = get_image_tensors(os.path.join(folder, 'images'))
        self.images = images

        self.sparse_points = None
        self.dense_points = None
        self.have_points = read_points
        self.batch_points = batch_points 
        if read_points:
            dense_path = os.path.join(folder, 'points3D.npy')
            dense_points = torch.FloatTensor(np.load(dense_path))
            dense_points = random_sample_points(dense_points, sample_rate)
            self.dense_points = dense_points

    def get_camera_centers(self):
        R, T = self.R, self.T 
        centers = torch.bmm(R, -T.unsqueeze(-1)).squeeze(-1) 
        return centers

    def __len__(self):
        return len(self.cameras)
    
    def __getitem__(self, index):
        if self.have_points:
            dense_n = self.dense_points.size(0)
            sample_idx = torch.rand(self.batch_points)
            sample_idx = (sample_idx * dense_n).long()
            points = self.dense_points[sample_idx]
        else: 
            points = torch.zeros(self.batch_points, 3)
    
        data = {
            'camera': self.cameras[index],
            'color': self.images[index],
            'points': points,
        }
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder')
    args = parser.parse_args()

    dataset = TanksAndTemplesDataset(args.folder, read_points=True, sample_rate=0.2)
    data = dataset[0]
    points = data['points']
    points_dense = dataset.dense_points
    print('dense points: {}'.format(points_dense.size()))
    points = vedo.Points(points_dense)
    vedo.show(points, axes=1)

if __name__ == '__main__':
    main()
