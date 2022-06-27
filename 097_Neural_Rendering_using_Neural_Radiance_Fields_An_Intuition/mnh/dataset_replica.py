import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from .utils_camera import get_ndc_grid
from .utils import get_image_tensors

def get_points_from_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[3:]
        points = []
        for line in lines:
            xyz_str = line.split(' ')[1:4]
            xyz= [float(s) for s in xyz_str]
            points.append(xyz)
        points = np.array(points)
        points = torch.FloatTensor(points)
        return points

def ndc_to_screen(intrinsic):
    W, H, fx, fy, px, py = intrinsic
    # convert from NDC space to screen space
    half_w, half_h = W/2, H/2
    fx_new = fx * half_w
    fy_new = fy * half_h
    px_new = -(px * half_w) + half_w
    py_new = -(py * half_h) + half_h
    return [W, H, fx_new, fy_new, px_new, py_new]
class ReplicaDataset(Dataset):
    def __init__(
        self,
        folder:str,
        focal_length:float=2.7778,
        read_points:bool=False,
        batch_points:int=10000
    ):
        R = np.load(os.path.join(folder, 'R.npy'))
        T = np.load(os.path.join(folder, 'T.npy'))
        R, T = torch.tensor(R), torch.tensor(T)
        self.R = R 
        self.T = T 
        self.focal_length = ((focal_length, focal_length),)
        self.principal_point = ((0, 0),)

        intrinsic = [512, 512, focal_length, focal_length, 0, 0]
        self.intrinsic = ndc_to_screen(intrinsic)  # W, H, fx, fy, px, py
        
        cameras = []
        for i in range(R.size(0)):
            cam  = PerspectiveCameras(
                focal_length=self.focal_length,
                principal_point=self.principal_point,
                R = R[i][None],
                T = T[i][None]
            )
            cameras.append(cam)
        self.cameras = cameras 

        images = get_image_tensors(os.path.join(folder, 'images'))
        depths = np.load(os.path.join(folder, 'depth.npy'))
        depths = torch.tensor(depths) #(N, h, w)
        self.images = images
        self.depths = depths

        self.dense_points = None
        self.have_points = read_points
        self.batch_points = batch_points 
        if read_points:
            dense_path = os.path.join(folder, 'dense/points3D.txt')
            dense_points = get_points_from_file(dense_path)
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
            'depth': self.depths[index],
            'points': points,
        }
        return data

def unproject_depth_points(depth, camera):
    '''
    Unproject depth points into world coordinates 
    '''
    # print(camera.get_full_projection_transform().get_matrix())
    size = list(depth.size())
    ndc_grid = get_ndc_grid(size).to(depth.device) #(h, w)
    ndc_grid[..., -1] = depth
    xy_depth = ndc_grid.view(1, -1, 3)
    points = camera.unproject_points(xy_depth)[0]
    return points

def dataset_to_depthpoints(dataset, point_num=None):
    '''
    Unproject all depth points within dataset into world coordinates 
    Args
        dataset
    Return
        points: (point_num, 3)
    '''
    points_all = []
    for i in range(len(dataset)):
        data = dataset[i]
        depth = data['depth']
        camera = data['camera']
        points = unproject_depth_points(depth, camera)
        points_all.append(points)
    
    points = torch.cat(points_all, dim=0)
    if point_num is not None:
        sample_idx = torch.randperm(points.size(0))[:point_num]
        points = points[sample_idx]
    return points
