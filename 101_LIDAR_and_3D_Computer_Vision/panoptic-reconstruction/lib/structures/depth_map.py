import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from lib.visualize.pointcloud import write_pointcloud


class DepthMap(object):
    def __init__(self, depth_map=None, intrinsic_matrix=None):
        if isinstance(depth_map, str):
            depth_map = torch.from_numpy(np.array(Image.open(depth_map))).float() / 1000.0
        self.depth_map = depth_map
        self.intrinsic_matrix = intrinsic_matrix

    def load_from(self, filename):
        depth_image = torch.from_numpy(np.array(Image.open(filename))).float()
        self.depth_map = depth_image / 1000.0

    def get_tensor(self):
        return self.depth_map.clone()

    def set_intrinsic(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix

    def get_intrinsic(self):
        return self.intrinsic_matrix.clone()

    def save(self, filename):
        plt.imsave(filename, self.depth_map.numpy(), cmap='rainbow')

    def mask_out(self, mask):
        self.depth_map = self.depth_map * mask

    def to_pointcloud(self, filename):
        pointcloud, _ = self.compute_pointcloud()
        write_pointcloud(pointcloud, None, filename)

    def to_pointcloud_with_colors(self, colors, filename):
        pointcloud, coords = self.compute_pointcloud()
        color_values = colors[coords[:, 0], coords[:, 1]]
        write_pointcloud(pointcloud, color_values, filename)

    def compute_pointcloud(self):
        coords2d = self.depth_map.nonzero(as_tuple=False)
        depth_map = self.depth_map[coords2d[:, 0], coords2d[:, 1]].reshape(-1)

        yv = coords2d[:, 0].reshape(-1).float() * depth_map.float()
        xv = coords2d[:, 1].reshape(-1).float() * depth_map.float()

        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float()])
        pointcloud = torch.mm(torch.inverse(self.intrinsic_matrix.float()), coords3d.float()).t()[:, :3]

        return pointcloud, coords2d

    def compute_normal(self):
        # linearize
        width = self.depth_map.shape[1]
        height = self.depth_map.shape[0]
        depth_map = self.depth_map.reshape(-1).float().cuda()

        yv, xv = torch.meshgrid([torch.arange(height),
                                 torch.arange(width)])

        yv = yv.reshape(-1).float().cuda() * depth_map.float()
        xv = xv.reshape(-1).float().cuda() * depth_map.float()
        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float().cuda()])
        pointcloud = torch.mm(torch.inverse(self.intrinsic_matrix.float().cuda()), coords3d.float()).t()[:, :3]

        '''
           MC
        CM-CC-CP
           PC
        '''
        output_normals = torch.zeros((3, height, width)).cuda()

        y, x = torch.meshgrid([torch.arange(1, height - 1),
                               torch.arange(1, width - 1)])
        y = y.reshape(-1)
        x = x.reshape(-1)

        CC = pointcloud[(y + 0) * width + (x + 0)]
        PC = pointcloud[(y + 1) * width + (x + 0)]
        CP = pointcloud[(y + 0) * width + (x + 1)]
        MC = pointcloud[(y - 1) * width + (x + 0)]
        CM = pointcloud[(y + 0) * width + (x - 1)]

        n = torch.cross(PC - MC, CP - CM).transpose(1, 0).cuda()
        l = torch.norm(n, dim=0).cuda()
        output_normals[:, y, x] = n / (-l)

        # filter1: zero_depth and their neighbouring
        zeros = (self.depth_map == 0).nonzero()
        output_normals[:, zeros[:, 0], zeros[:, 1]] = 0

        zeros_height_lower = zeros.clone()
        zeros_height_lower[:, 0] -= 1
        zeros_height_lower[:, 0] = torch.clamp(zeros_height_lower[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_lower[:, 0], zeros_height_lower[:, 1]] = 0

        zeros_height_upper = zeros.clone()
        zeros_height_upper[:, 0] += 1
        zeros_height_upper[:, 0] = torch.clamp(zeros_height_upper[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_upper[:, 0], zeros_height_upper[:, 1]] = 0

        zeros_width_lower = zeros.clone()
        zeros_width_lower[:, 1] -= 1
        zeros_width_lower[:, 1] = torch.clamp(zeros_width_lower[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_lower[:, 0], zeros_width_lower[:, 1]] = 0

        zeros_width_upper = zeros.clone()
        zeros_width_upper[:, 1] += 1
        zeros_width_upper[:, 1] = torch.clamp(zeros_width_upper[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_upper[:, 0], zeros_width_upper[:, 1]] = 0

        return output_normals
