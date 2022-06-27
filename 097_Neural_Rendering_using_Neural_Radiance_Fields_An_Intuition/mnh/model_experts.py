from typing import Tuple
import math
import torch
import torch.nn as nn
from .utils_camera import *
from .utils_model import *
from .plane_geometry import PlaneGeometry
from .implicit_experts import NerfExperts
from .utils import *

class ModelExperts(nn.Module):
    def __init__(
        self,
        n_plane: int,
        image_size: Tuple[int],
        # Radiance field 
        n_harmonic_functions_pos:int,
        n_harmonic_functions_dir:int,
        n_hidden_neurons_pos:int,
        n_hidden_neurons_dir:int,
        n_layers:int,
        # train & test
        n_train_sample:int, 
        n_infer_sample:int, 
        anti_aliasing:bool,
        premultiply_alpha: bool,
        # accelerate
        n_bake_sample:int,
        bake_res:int,
        filter_thresh:float,
        white_bg:bool
    ):
        super().__init__()
        self.n_plane = n_plane
        self.plane_geo = PlaneGeometry(n_plane)
        self.image_size = image_size
        self.ndc_grid = get_ndc_grid(image_size)

        self.plane_radiance_field = NerfExperts(
            n_harmonic_functions_pos,
            n_harmonic_functions_dir,
            n_hidden_neurons_pos,
            n_hidden_neurons_dir,
            n_layers,
            n_experts=n_plane
        )

        self.n_train_sample = n_train_sample 
        self.n_infer_sample = n_infer_sample
        self.anti_aliasing = anti_aliasing
        self.premultiply_alpha = premultiply_alpha

        self.planes_alpha = None
        self.n_bake_sample = n_bake_sample
        self.bake_res = bake_res
        self.filter_thresh = filter_thresh 
        self.white_bg = white_bg

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def bake_planes_alpha(self):
        resolution = self.bake_res
        planes_points = self.plane_geo.get_planes_points(resolution) #(plane_n, res, res, 3)
        planes_points = planes_points.view(-1, 3)
        planes_idx = torch.arange(self.n_plane, device=planes_points.device)
        planes_idx = planes_idx.view(-1, 1, 1).repeat(1, resolution, resolution).view(-1)

        points_total_n = (resolution ** 2) * self.n_plane
        sample_n = self.n_bake_sample
        chunk_n = math.ceil(points_total_n / sample_n)
        planes_alpha = []
        with torch.no_grad():
            for i in range(chunk_n):
                start = i * sample_n
                end = min((i+1)*sample_n, points_total_n)
                points = planes_points[start:end,:]
                dirs = torch.zeros_like(points)
                idx = planes_idx[start:end]
                rgba = self.plane_radiance_field(points, dirs, idx)
                rgba = rgba.detach()
                alpha = rgba[..., -1]
                planes_alpha.append(alpha)
        
        planes_alpha = torch.cat(planes_alpha, dim=0)
        planes_alpha = planes_alpha.view(self.n_plane, 1, resolution, resolution)
        self.planes_alpha = planes_alpha #(plane_n, 1, res, res)
        torch.cuda.empty_cache()
        print('Baked planes alpha as [{} * {}]'.format(resolution, resolution))

    def ray_plane_intersect(self, camera, ndc_points):
        '''
        Return
            world_points: (plane_n, point_n, 3)
            planes_depth:  (plane_n, point_n)
            hit:          (plane_n, point_n)
        '''
        planes_basis = self.plane_geo.basis()
        planes_center = self.plane_geo.position() #(plane_n, 3)
        planes_depth, world_points = ray_plane_intersection(
            planes_basis,
            planes_center,
            camera, 
            ndc_points
        )

        xy_basis = planes_basis[:, :, :2] #(plane_n, 3, 2)
        planes_points = torch.bmm(world_points - planes_center.unsqueeze(1), xy_basis) #(plane_n, point_n, 2)
        
        in_planes = check_inside_planes(planes_points, self.plane_geo.size()) #(plane_n, point_n)
        hit = torch.logical_and(in_planes, planes_depth > 0) #(plane_n, point_n)
        return world_points, planes_points, planes_depth, hit

    def sort_depth_index(self, planes_depth):
        '''
        sort points along ray with depth to planes 
        Args
            planes_depth: (plane_n, point_n)
        Return
            sort_id_0, sort_id_0
        '''
        depth_sorted, sort_id_0 = torch.sort(planes_depth, dim=0, descending=False) # ascending
        point_n = planes_depth.size(1)
        sort_id_1 = torch.arange(point_n)[None].to(sort_id_0.device)
        sort_idx = [sort_id_0, sort_id_1]
        return depth_sorted, sort_idx

    def get_planes_indices(self, hit):
        '''
        Return
            planes_idx_full: (plane_n, point_n) -> accending in 1-dim
        '''
        plane_n, point_n = hit.shape
        planes_idx_full = torch.arange(plane_n, device=hit.device)
        planes_idx_full = planes_idx_full.unsqueeze(1).repeat(1, point_n)
        return planes_idx_full

    def predict_points_rgba_experts(self, camera, points, planes_idx):
        '''
        Args
            camera
            points: (hit_n, 3): in world coord.
            planes_idx: (hit_n) 
        Return
            poins_rgba: (hit_n, 4)
        '''
        view_dirs = get_normalized_direction(camera, points) #(b, 3)
        points_rgba = self.plane_radiance_field(points, view_dirs, planes_idx)
        return points_rgba

    def alpha_composite(self, rgb, alpha, depth):
        '''
        Return
            color: (point_n, 3)
            depth: (point_n)
        '''
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        depth = torch.sum(depth * alpha_weight, dim=0)   
        color = torch.sum(rgb * alpha_weight.unsqueeze(-1), dim=0) #(piont_n, 3)
        
        if self.white_bg:
            alpha_sum = torch.sum(alpha_weight, dim=0).unsqueeze(-1) #(point_n, 1)
            white = torch.ones_like(color) #(point_n, 3)
            color = color + (1 - alpha_sum) * white
        return color, depth

    def no_hit_output(self, ndc_points):
        if self.white_bg:
            color_bg = torch.ones_like(ndc_points)
        else:
            color_bg = torch.zeros_like(ndc_points)
        
        point_n = ndc_points.size(0)
        device = ndc_points.device 
        dummy_output = {
            'color': color_bg, #(point_n, 3)
            'depth': torch.zeros(point_n, device=device), #(point_n, )
            # 'points': torch.zeros(point_n, 3, device=device), #(point_n, 3)
            # 'eval_num': torch.zeros(point_n, device=device), #(point_n, )
        }
        return dummy_output

    def sample_baked_alpha(self, planes_points, hit):
        '''
        Return 
            alpha_sample: (plane_n, point_n)
        '''
        alpha_sample = grid_sample_planes(
            sample_points=planes_points, 
            planes_wh=self.plane_geo.size(),
            planes_content=self.planes_alpha,
            mode='nearest',
            padding_mode='border'
        ) # (plane_n, point_n, 1)
        alpha_sample = alpha_sample.squeeze(-1)
        alpha_sample[hit==False] = 0
        return alpha_sample

    def process_ndc_points(self, 
        camera, 
        ndc_points
    ):
        '''
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        '''
        world_points, _, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        
        planes_idx_full = self.get_planes_indices(hit)
        planes_idx = planes_idx_full[hit]
        points = world_points[hit]
        points_rgba = self.predict_points_rgba_experts(camera, points, planes_idx)
        rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        rgba[hit] = points_rgba

        depth, sort_idx = self.sort_depth_index(planes_depth)
        rgba = rgba[sort_idx]
        rgb, alpha = rgba[:,:,:-1], rgba[:,:,-1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
        
        # compute unprojected points with predicted depth
        # points = unproject_points(camera, ndc_points, depth)
        # points = points.squeeze(0).detach()       
        # eval_num = torch.sum(hit, dim=0).float() #(point_n)

        output = {
            'color': color, #(point_3, 3)
            'depth': depth, #(point_3, )
            # 'points': points, #(point_3, 3)
            # 'eval_num': eval_num, #(piont_3, )
        }
        return output

    def process_ndc_points_with_alpha(self, 
        camera, 
        ndc_points
    ):
        '''
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        '''
        world_points, planes_points, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        
        alpha_baked = self.sample_baked_alpha(planes_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        alpha = alpha_baked[sort_idx] #(plane_n, point_n)
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        contrib = alpha_weight > self.filter_thresh
        hit = hit[sort_idx] #(plane_n, point_n)
        hit = torch.logical_and(hit, contrib)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        
        world_points = world_points[sort_idx] #(plane_n, point_n, 3)
        points = world_points[hit]
        planes_idx_full = self.get_planes_indices(hit)
        planes_idx_sorted = planes_idx_full[sort_idx]
        planes_idx = planes_idx_sorted[hit]
        points_rgba = self.predict_points_rgba_experts(camera, points, planes_idx)
        rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        rgba[hit] = points_rgba
        
        # rgb = rgba[:,:,:-1] # use sampled alpha
        rgb, alpha = rgba[:,:,:-1], rgba[:,:,-1] # use original alpha
        color, depth = self.alpha_composite(rgb, alpha, depth)
        # eval_num = torch.sum(hit, dim=0).float() #(point_n)

        output = {
            'color': color, #(s, 3)
            'depth': depth, #(s, )
            # 'eval_num': eval_num, #(s,)
        }
        return output  

    def process(self, camera, ndc_points):
        out = None
        if self.planes_alpha is not None:
            out = self.process_ndc_points_with_alpha(camera, ndc_points)
        else:
            out = self.process_ndc_points(camera, ndc_points)
        return out 

    def ndc_points_full(self, camera):
        '''
        Return:
            NDC points: (img_h*img_w, 3)
        '''
        device = camera.device
        self.ndc_grid = self.ndc_grid.to(device)
        ndc_grid = self.ndc_grid.clone()
        if self.training and self.anti_aliasing:
            ndc_grid = oscillate_ndc_grid(ndc_grid)
        ndc_points = ndc_grid.view(-1, 3) #(img_h*img_w, 3)
        return ndc_points

    def forward_train(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        # sample_idx = torch.randperm(img_pixel_num)[:self.n_train_sample] #(n_train_sample, )
        sample_idx = torch.rand(self.n_train_sample)
        sample_idx = (sample_idx * img_pixel_num).long()
        ndc_points = ndc_points_full[sample_idx] #(n_train_sample, 3)
        output = self.process(camera, ndc_points)
        output['sample_idx'] = sample_idx
        return output

    def forward_full_image(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        if self.n_infer_sample > 0:
            sample_num = self.n_infer_sample
        else:
            sample_num = img_pixel_num
        
        chunk_num = math.ceil(img_pixel_num / sample_num)
        chunk_outputs = []
        for i in range(chunk_num):
            start = i * sample_num
            end = min((i+1) * sample_num, img_pixel_num)
            ndc_points = ndc_points_full[start:end]
            chunk_out = self.process(camera, ndc_points)
            chunk_outputs.append(chunk_out)
        
        # aggregate output from all chunks
        img_wh = self.image_size
        shapes = {
            'color': [*img_wh, 3],
            'depth': [*img_wh],
            # 'points': [-1, 3],
            # 'eval_num': [-1]
        }
        output = {
            key: torch.cat([
                chunk_out[key] for chunk_out in chunk_outputs
            ], dim=0).view(*shape)
            for key, shape in shapes.items()
        }
        return output 

    def forward(self, camera):
        ndc_points_full = self.ndc_points_full(camera)

        if self.training: 
            output = self.forward_train(camera, ndc_points_full)
        else: 
            output = self.forward_full_image(camera, ndc_points_full)
        return output

def test_forward():
    from pytorch3d.renderer import PerspectiveCameras
    device = torch.device('cuda:0')

    model = ModelExperts(
        n_plane=10,
        image_size=[200, 200],
        n_harmonic_functions_pos=10,
        n_harmonic_functions_dir=4,
        n_hidden_neurons_pos=32,
        n_hidden_neurons_dir=32,
        n_layers=2,
        # train & test
        n_train_sample=4096, 
        n_infer_sample=1024, 
        anti_aliasing=True,
        premultiply_alpha=False,
        # accelerate
        n_bake_sample=1024,
        bake_res=200,
        filter_thresh=0.01,
        white_bg=False
    ).to(device)

    camera = PerspectiveCameras(
        focal_length=((1.0,1.0),),
        principal_point=((0.0,0.0),),
        R=torch.eye(3)[None],
        T=torch.zeros(1, 3),
    ).to(device)

    model.eval()
    model.bake_planes_alpha()
    out = model(camera)
    image = out['color']
    print('Output image size: {}'.format(tuple(image.size())))


if __name__ == '__main__':
    test_forward()
