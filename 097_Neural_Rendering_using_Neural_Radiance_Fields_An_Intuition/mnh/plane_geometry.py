import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PlaneGeometry(nn.Module):
    def __init__(
        self,
        n_plane:int
    ):
        super().__init__()
        self.n_plane = n_plane
        self.center = nn.Parameter(torch.FloatTensor(n_plane, 3))
        self.xy = nn.Parameter(torch.FloatTensor(n_plane, 3, 2))
        # self.yz = nn.Parameter(torch.FloatTensor(n_plane, 3, 2))
        self.wh = nn.Parameter(torch.FloatTensor(n_plane, 2))

        self.center.data.uniform_(0, 1)
        self.wh.data[:] = 1
        eyes = torch.eye(3)[None].repeat(n_plane, 1, 1)
        self.xy.data = eyes[:,:,:2]
        # self.yz.data = eyes[:,:,:2]

        self.init_with_box = False # check if the initialization including boxes

    def initialize(self, 
        points, 
        lrf_neighbors:int=50,
        wh:float=1.0
    ):
        '''
        Initialize planes
            -position: FPS points
            -roation: local PCA basis
            -size: specified in args
        '''
        sample_idx, center = farthest_point_sample(points, self.n_plane)
        lrf = get_points_lrf(points, neighbor_num=lrf_neighbors, indices=sample_idx) #(point_n, 3, 3)
        self.center.data = center
        self.xy.data = lrf[:,:,:2]
        # self.yz.data = lrf[:,:,1:]
        self.wh.data[:] = wh

    def initialize_with_box(
        self,
        points, # (p_n, 3)
        lrf_neighbors:int,  
        wh:float,
        box_factor:float=1.5,
        random_rate:float=0.0
    ):  
        device = points.device
        mean = torch.mean(points, dim=0)
        bound_max = torch.max(points - mean, dim=0)[0] * box_factor + mean 
        bound_min = torch.min(points - mean, dim=0)[0] * box_factor + mean
        box_len = torch.max(bound_max - bound_min)
        x_max, y_max, z_max = bound_max 
        x_min, y_min, z_min = bound_min
        x_mid, y_mid, z_mid = (x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2 
        face_centers = torch.FloatTensor([
            [x_min, y_mid, z_mid],
            [x_max, y_mid, z_mid],
            [x_mid, y_min, z_mid],
            [x_mid, y_max, z_mid],
            [x_mid, y_mid, z_min],
            [x_mid, y_mid, z_max]
        ]).to(device)
        eye = torch.eye(3).to(device)
        face_xy = torch.stack([
            eye[:,[1, 2]],
            eye[:,[1, 2]],
            eye[:,[0, 2]],
            eye[:,[0, 2]],
            eye[:,[0, 1]],
            eye[:,[0, 1]],
        ], dim=0)

        face_n = 6 
        sample_n = self.n_plane - face_n
        if random_rate > 0:
            rand_n = int(sample_n * random_rate)
            fps_n = sample_n - rand_n
            rand_idx = torch.randperm(points.size(0))[:rand_n]
            rand_center = points[rand_idx]
            fps_idx, fps_center = farthest_point_sample(points, fps_n)
            sample_idx = torch.cat([rand_idx, fps_idx])
            center = torch.cat([rand_center, fps_center], dim=0)
        else:
            sample_idx, center = farthest_point_sample(points, sample_n)
        lrf = get_points_lrf(points, neighbor_num=lrf_neighbors, indices=sample_idx) #(point_n, 3, 3)
        
        self.center.data = torch.cat([face_centers, center], dim=0)
        self.xy.data = torch.cat([face_xy, lrf[:,:,:2]], dim=0)
        self.wh.data[:face_n] = box_len
        self.wh.data[face_n:] = wh

        self.init_with_box = True

    def position(self):
        return self.center

    def basis(self):
        basis = orthonormal_basis_from_xy(self.xy)
        # basis = orthonormal_basis_from_yz(self.yz)
        return basis

    def size(self):
        return self.wh

    def get_planes_points(self, resolution:int):
        '''
        Get the the position of plane points (image pixel) with resolution in H, W
        Args
            resolution 
        Return
            plane points in world coordinate
            (n_plane, res, res, 3) 
        '''
        device = self.center.device 
        pix_max =  0.5 * (1 - 0.5/resolution)
        pix_min = -0.5 * (1 - 0.5/resolution)
        stride = torch.linspace(pix_max, pix_min, resolution, device=device)
        plane_xy = torch.stack(torch.meshgrid(stride, stride), dim=-1) #(res, res, 2)
        plane_xy = torch.flip(plane_xy, dims=[-1]) #(res, res, 2), last dim=(x, y) -> in (w, h) direction
        

        planes_xy = plane_xy.view(1, -1, 2).repeat(self.n_plane, 1, 1) #(plane_n, res*res, 2)
        planes_xy = planes_xy * self.wh.unsqueeze(1) 
        basis = self.basis() #(plane_n, 3, 3)
        basis_xy = basis[:, :, :-1] #(plane_n, 3, 2)

        from_center = torch.bmm(planes_xy, basis_xy.transpose(1, 2)) #(plane_n, res*res, 3)
        planes_points = self.center.unsqueeze(1) + from_center
        planes_points = planes_points.view(self.n_plane, resolution, resolution, 3)
        return planes_points

    def sample_planes_points(self, points_n):
        '''
        Sample random points on planes, total number <= points_n
        Return 
            planes_points: (plane_n*sample_n, 3)
            planes_idx: (plane_n*sample_n, )
        '''
        device = self.center.device
        sample_n = math.ceil(points_n / self.n_plane) #sample number per plane
        sample_uv = torch.rand(sample_n, 2, device=device) - 0.5 #(sample_n, 2)
        sample_coord = torch.einsum('pd,sd->psd', self.wh, sample_uv) #(planes_n, sample_n, 2)
        basis = self.basis() #(plane_n, 3, 3)
        xy = basis[:,:,:2] #(plane_n, 3, 2)
        world_coord = torch.einsum('psa,pba->psb', sample_coord, xy) #(planes_n, sample_n, 3)
        planes_points = self.center.unsqueeze(1) + world_coord # (plane_n, sample_n, 3)        
        planes_points = planes_points.detach()

        planes_idx = torch.arange(self.n_plane, device=device)
        planes_idx = planes_idx.unsqueeze(1).repeat(1, sample_n)
        
        planes_points = planes_points.view(-1, 3)
        planes_idx = planes_idx.view(-1)
        return planes_points, planes_idx

    def planes_vertices(self):
        '''
        Return
            planes_vertices: (plane_n, 4, 3)
            which are 4 corners of each planes
        '''
        center = self.center #(palne_n, 3)
        wh = self.wh  #(plane_n, 2)
        basis = self.basis() 
        xy_basis = basis[:,:,:2] #(plane_n, 3, 2)

        xy_vec = xy_basis * wh.unsqueeze(1) # length of width & height
        x_vec, y_vec = xy_vec[:,:,0], xy_vec[:,:,1]
        planes_vertices = []
        for i_x in [-0.5, 0.5]:
            for i_y in [-0.5, 0.5]:
                vertices = center + i_x * x_vec + i_y * y_vec
                planes_vertices.append(vertices)
        planes_vertices = torch.stack(planes_vertices, dim=1)

        return planes_vertices.detach()

    def forward(self, points):
        '''
        Input:
            points: (point_num, 3) xyz
        Return:
            loss: point-plane "closeness"
        '''
        # compute orthonormal basis
        xyz = self.basis()
        x, y, z = xyz[: ,: ,0], xyz[: ,: ,1], xyz[:, :, 2]
        
        diff = points.unsqueeze(0) - self.center.unsqueeze(1) #(plane_n, point_n, 3)
        dist_x = torch.bmm(diff, x.unsqueeze(-1)).squeeze() #(plane_n, point_n)
        dist_y = torch.bmm(diff, y.unsqueeze(-1)).squeeze()
        dist_z = torch.bmm(diff, z.unsqueeze(-1)).squeeze()  
        
        dist_x = torch.abs(dist_x) - self.wh[:, 0].unsqueeze(-1)/2
        dist_x = torch.clamp(dist_x, min=0)
        dist_y = torch.abs(dist_y) - self.wh[:, 1].unsqueeze(-1)/2
        dist_y = torch.clamp(dist_y, min=0)
        distance = dist_x**2 + dist_y**2 + dist_z**2

        min_dist, min_id = torch.min(distance, dim=0) #(point_n)
        loss_point2plane = torch.mean(min_dist)
       
        if self.init_with_box:
            face_n = 6 
            loss_area = torch.mean(torch.abs(self.wh[face_n:, 0] * self.wh[face_n:, 1]))
        else:
            loss_area = torch.mean(torch.abs(self.wh[:, 0] * self.wh[:, 1]))

        output = {
            'loss_point2plane': loss_point2plane,
            'loss_area': loss_area
        }
        return output

def farthest_point_sample(
    points,
    sample_n: int
):
    '''
    Input:
        points: (point_n, dim)
    Return:
        idx: (sample_n)
        points_sample: (sample_n, dim)
    '''
    idx = 0
    sample_set= [idx]
    dist2set = torch.tensor([]).to(points.device)
    for i in range(sample_n - 1):
        dist = points - points[idx]
        dist = torch.sum(dist**2, dim=1)[:, None]
        dist2set = torch.cat([dist2set, dist], dim=1)
        min_dist, _ = torch.min(dist2set, dim=1) #(point_n,)
        _, max_id = torch.max(min_dist, dim=0)
        idx = max_id.item()
        sample_set.append(idx) 

    points_sample = points[sample_set]
    sample_set = torch.LongTensor(sample_set)
    return sample_set, points_sample

def get_points_lrf(
    points,
    neighbor_num:int,
    indices,
    chunk_size:int=200
):
    '''
    Input:
        points: (point_n, 3)
        indices: (sample_n,) index of partial points -> reduce computation
    Output:
        Local reference frame at each point computed by PCA
        lrf: (point_n, 3, 3) basis are aranged in columns
    '''
    samples = points[indices] #(sample_n, 3)
    dist = samples.unsqueeze(1) - points.unsqueeze(0) #(s, p, 3)
    dist = torch.sum(dist**2, dim=-1) #(s, p)
    dist_n, neighbor_idx = torch.topk(dist, k=neighbor_num, dim=-1, largest=False)
    neighbors = points[neighbor_idx].cpu() #(s, n, 3)
    lrf_list = []
    sample_n = samples.size(0)
    chunk_n = math.ceil(sample_n/chunk_size)
    for i in range(chunk_n):
        start = i * chunk_size 
        end = min((i+1)*chunk_size, sample_n)
        U, S, V_t = torch.pca_lowrank(neighbors[start:end])
        lrf_list.append(V_t)
        # U:(s, n, n), S:(s, min(n,3)), V_t:(s, 3, 3)
    lrf = torch.cat(lrf_list, dim=0).to(points.device)
    return lrf

def orthonormal_basis_from_xy(xy):
    '''
    compute orthonormal basis from xy vector: (n, 3, 2)
    '''
    x, y = xy[:,:,0], xy[:,:,1]
    z = torch.cross(x, y, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    xyz = torch.stack([x,y,z], dim=-1)
    return xyz

def orthonormal_basis_from_yz(yz):
    '''
    compute orthonormal basis from yz vector: (n, 3, 2)
    '''
    y, z = yz[:,:,0], yz[:,:,1]
    x = torch.cross(y, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    xyz = torch.stack([x,y,z], dim=-1)
    return xyz

def test(): 
    from .utils_vedo import visualize_geometry
    points = torch.randn(1000, 3)
    model = PlaneGeometry(10)
    model.initialize(points, 20)
    planes_points, planes_idx = model.sample_planes_points(2000)
    
    visualize_geometry(planes_points, model)

if __name__ == '__main__':
    test()