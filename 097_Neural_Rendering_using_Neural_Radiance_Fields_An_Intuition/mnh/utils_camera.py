from typing import Tuple
import torch
import torch.nn.functional as F
from .utils_model import detect_invalid_values

def get_ndc_grid(
    image_size: Tuple[int, int]
):  
    '''
    Get the NDC coordinates of every pixel
    This follows the pytorch3d module NDCGridRaysampler(GridRaysampler)
    here the x is along horizontal direction (width), 
        and y is along vertical (height)
    
    Args
        image_size = (height, width)
    Return
        ndc_girds: in shape (height, width, 3), each position is (x, y, 1)
    '''
    height, width = image_size
    half_pix_width = 1.0 / width
    half_pix_height = 1.0 / height
    min_x =  1.0 - half_pix_width
    max_x = -1.0 + half_pix_width
    min_y =  1.0 - half_pix_height
    max_y = -1.0 + half_pix_height
        
    x_grid_coord = torch.linspace(min_x, max_x, width, dtype = torch.float32)
    y_grid_coord = torch.linspace(min_y, max_y, height, dtype = torch.float32)
    yy, xx = torch.meshgrid(y_grid_coord, x_grid_coord)
    xy_grid = torch.stack([xx, yy], dim=-1) #(h, w, 2)
    ndc_grid = torch.cat([xy_grid, torch.ones(height, width, 1)], dim=-1)
    return ndc_grid

def oscillate_ndc_grid(ndc_grid):
    '''
    oscillate NDC girds within pixel w/h when trainig -> anti-aliasing
    Args & Return:
        ndc_grid: (h, w, 3)
    '''
    h, w, _ = ndc_grid.size()
    device = ndc_grid.device
    half_pix_w = 1.0 / w
    half_pix_h = 1.0 / h
    noise_w = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_w
    noise_h = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_h
    ndc_grid[:,:,0] += noise_w
    ndc_grid[:,:,1] += noise_h
    return ndc_grid

def filter_tiny_values(tensor, eps:float=1e-5):
    tensor_sign = torch.sign(tensor)
    tensor_sign[tensor_sign == 0] = 1
    tensor_abs = torch.clamp(torch.abs(tensor), min=eps)
    tensor_out = tensor_sign * tensor_abs 
    return tensor_out

def get_camera_k(camera):
    '''
    k = [
        [fx,  0, 0],
        [0,  fy, 0],
        [px, py, 1],
    ]
    '''
    proj_trans = camera.get_projection_transform()
    proj_mat = proj_trans.get_matrix().squeeze() #(4, 4)
    k = proj_mat[:3, :3]
    k[-1, -1] = 1
    return k

def get_camera_k_inv(camera):
    k = get_camera_k(camera)
    return k.inverse()

def get_world2cam(camera):
    R = camera.R[0]
    T = camera.T[0]
    trans = R.new_zeros(4, 4)
    trans[:3, :3] = R 
    trans[-1, :3] = T
    trans[-1, -1] = 1
    return trans 

def get_cam2world(camera):
    R = camera.R[0]
    T = camera.T[0]
    trans = R.new_zeros(4, 4)
    trans[:3, :3] = R.t()
    trans[-1, :3] = torch.matmul(R, -T.unsqueeze(-1)).squeeze(-1)
    trans[-1, -1] = 1
    return trans 

def get_camera_center(camera):
    '''
    Return
        center: (1, 3)
    '''
    # R = camera.R[0]
    # T = camera.T[0]
    # center = torch.matmul(R, -T.unsqueeze(-1)).squeeze(-1)
    # center = center[None]
    center = camera.get_camera_center()
    return center 

def camera_ray_directions(camera, ndc_points: torch.Tensor):
    '''
    Calculate (x/z, y/z, 1) of each NDC points, under camera coord.
    Args
        ndc_points: (point_n, 3)
    Return
        xy1_points: (point_n, 3)
    '''
    device = ndc_points.device
    k = get_camera_k(camera)
    fx, fy = k[0, 0], k[1, 1]
    px, py = k[2, 0], k[2, 1]
    shift = torch.tensor([[-px, -py]]).to(device) #(1, 2) (-p_x, -p_y) for PerspectiveCam, (-w_1,-h_1) for FoVCam
    scale = torch.tensor([[fx, fy]]).to(device)  #(1 ,2)  (f_x, f_y) for PerspectiveCam,  (s_1, s_2) for FoVCam 
    xy1_points = torch.ones_like(ndc_points) # (point_n, 3)
    xy1_points[:, :2] = (ndc_points[:, :2] + shift) / scale
    return xy1_points

def ray_plane_intersection(
    planes_frame: torch.Tensor,
    planes_center: torch.Tensor, 
    camera,
    ndc_points: torch.Tensor,
    eps:float=1e-5
):
    '''
    Calculate ray-plane intersection in world coord. (follow the paper formulation)
    Return
        depth: (plane_n, point_n)
        intersections: (plane_n, point_n, 3)
    '''
    normal_planes = planes_frame[:,:,-1] #(plane_n, 3)
    center_planes = planes_center # (plane_n, 3)
    R_cam = camera.R # (1, 3, 3)
    center_cam = get_camera_center(camera) # (1, 3)
    
    xy1_points = camera_ray_directions(camera, ndc_points)
    cam2world = R_cam[0].T
    d = xy1_points @ cam2world #(point_n, 3) dir in world coord.

    num = torch.sum((center_planes - center_cam) * normal_planes, dim=-1) #(plane_n,)
    den = torch.mm(normal_planes, d.T) #(plane_n, point_n)
    den = filter_tiny_values(den, eps)
    t = num.unsqueeze(-1)/den #(plane_n, point_n)
    td = t.unsqueeze(-1) * d.unsqueeze(0) #(plane_n, point_n, 3)
    o = center_cam.unsqueeze(0) #(1, 1, 3)
    intersections = o + td 
    depth = t
    return depth, intersections

def ray_plane_intersect_mt(planes_vertices, camera, ndc_points):
    '''
    Follow the formulation in: 
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    Return
        depth: (plane_n, point_n)
        intersections: (plane_n, point_n, 3)
    '''
    plane_n = planes_vertices.size(0)
    point_n = ndc_points.size(0)
    cam_center = get_camera_center(camera) # (1, 3)
    xy1 = camera_ray_directions(camera, ndc_points) #(point_n, 3)
    cam_R = camera.R
    cam2world = cam_R[0].T
    dirs = xy1 @ cam2world

    D = dirs #(point_n, 3)
    O = cam_center #(1, 3)
    A, B, C = planes_vertices[:,0], planes_vertices[:,1], planes_vertices[:,2] #(plane_n, 3)
    T  = O - A 
    E1 = B - A
    E2 = C - A 

    # prepare for each pairs, in shape (plane_n, point_n, 3)
    D  = D.unsqueeze(0).repeat(plane_n, 1, 1)
    T  = T.unsqueeze(1).repeat(1, point_n, 1)
    E1 = E1.unsqueeze(1).repeat(1, point_n, 1)
    E2 = E2.unsqueeze(1).repeat(1, point_n, 1)
    P  = torch.cross(D, E2)
    Q  = torch.cross(T, E1)

    det = filter_tiny_values(torch.sum(P * E1, dim=-1))
    det_inv = 1 / det
    t = torch.sum(Q * E2, dim=-1) * det_inv
    u = torch.sum(P * T,  dim=-1) * det_inv
    v = torch.sum(Q * D,  dim=-1) * det_inv

    u_inside = torch.logical_and(u >= 0, u <= 1)
    v_inside = torch.logical_and(v >= 0, v <= 1)
    hit = torch.logical_and(u_inside, v_inside)
    hit = torch.logical_and(hit, t >= 0)
    return t

def test_intersect():
    from .plane_geometry import PlaneGeometry
    from pytorch3d.renderer import PerspectiveCameras

    points = torch.randn(1000, 3).cuda()
    planes = PlaneGeometry(1000).cuda()
    planes.initialize(points)

    planes_center = planes.position() # (plane_n, 3)
    planes_R = planes.basis()     # (plane_n, 3, 3)
    planes_vertices = planes.planes_vertices() # (plane_n, 4, 3)

    ndc_grid = get_ndc_grid([100, 100]).cuda()
    ndc_points = ndc_grid.view(-1, 3)

    camera = PerspectiveCameras(
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        R=torch.eye(3)[None],
        T=torch.randn(1, 3)
    ).to('cuda')

    t1, intersect1 = ray_plane_intersection(planes_R, planes_center, camera, ndc_points)
    t2 = ray_plane_intersect_mt(planes_vertices, camera, ndc_points)
    print(torch.dist(t1, t2))
    diff = t1 - t2
    print('Diff. Max. = {}'.format(diff.abs().max().item()))

def get_depth_on_planes(
    planes_frame: torch.Tensor,
    planes_center: torch.Tensor, 
    camera,
    ndc_points: torch.Tensor,
    eps:float=1e-5
):  
    '''
    Get the depth value of each position in NDC space, 
    which corresponds to the projected point on the plane
    
    Args:
        planes_frame: (plane_n, 3, 3)
        planes_center: (plane_n, 3)
        camera: PerspectiveCameras(SfM), NoVPerspectiveCameras(OpenGL)
        ndc_points: (sample_n, 3)
    Return
        depths: (sample_n, plane_n)
    '''
    xy1_points = camera_ray_directions(camera, ndc_points)

    # all under [world coordinate]
    R_planes = planes_frame # (plane_n, 3,3) 
    center_planes = planes_center # (plane_n, 3)
    R_cam =  camera.R # (1, 3, 3)
    center_cam = get_camera_center(camera) # (1, 3)

    # plane normal(z-axis), center under [camera view coordinate]
    R_cam2planes = torch.matmul(R_cam.transpose(1, 2), R_planes) # (plane_n, 3, 3)
    normal_planes = R_cam2planes[:, :, -1] # (plane_n, 3) 
    translation = center_planes - center_cam # (plane_n, 3)
    center_planes = torch.matmul(translation.unsqueeze(1), R_cam).squeeze(1) # (plane_n, 3)

    numerator = torch.matmul(normal_planes.unsqueeze(1), center_planes.unsqueeze(2)) # (plane_n, 1, 1)
    numerator = numerator.squeeze(-1) # (plane_n, 1)
    denominator = torch.matmul(normal_planes, xy1_points.T) # (plane_n, sample_n)
    denominator = filter_tiny_values(denominator, eps)
    depth = numerator / denominator # (plane_n, sample_n)
    return depth

def get_transform_matrix(rotation, position):
    '''
    Args
        rotation: (n, 3, 3) rotation matrix
        poistion: (n, 3) pos in world coord.
    Return
        transform matrix (n, 4, 4) world -> object/camera
    '''
    R = rotation
    T = torch.matmul(-position.unsqueeze(1), R).squeeze(1)
    n = R.size(0)
    trans = torch.zeros(n, 4, 4).to(R.device)
    trans[:, :3, :3] = R
    trans[:, -1, :3] = T
    trans[:, -1, -1] = 1
    return trans

def transform_points_batch(points, transform, eps:float=1e-5):
    '''
    Transform batched points under homogeneous coordinates
    Args
        points: (B, N, 3)
        transform: (B, 4, 4)
    return 
        points_transformed: (B, N, 3)
    '''
    ones = torch.ones_like(points[...,0]).unsqueeze(-1)
    points_h = torch.cat([points, ones], dim=-1) # (B, N, 4)
    points_h_transformed = torch.matmul(points_h, transform)
    xyz = points_h_transformed[...,:-1]
    h = points_h_transformed[...,-1].unsqueeze(-1)    
    h = filter_tiny_values(h, eps)
    points_transformed = xyz / h
    return points_transformed

def rotate_translate(xyz, transform):
    '''
    Transform batched points with rotation + translation
    Args
        xyz: (B, N, 3)
        transform: (B, 4, 4) -> only R & T
    return 
        xyz_transformed: (B, N, 3)
    '''
    ones = torch.ones_like(xyz[...,0:1])
    xyz_1 = torch.cat([xyz, ones], dim=-1) # (B, N, 4)
    xyz_transformed = torch.matmul(xyz_1, transform[...,:3])
    return xyz_transformed

def unproject_points_pt3d(
    camera,
    xy_depth,
    world_coordinates=True
):
    '''
    pytorch3d camera.unproject_points() generate nan, inf values in our methods
    Args
        xy_depth: (..., 3) NDC points with depth
    Return
        unprojected points in world coordinates if world_coordinates==True(in camera coordinates otherwise)
    '''
    depth = xy_depth[..., -1]
    depth = filter_tiny_values(depth, eps=1e-7)
    K_matrix = camera.get_projection_transform().get_matrix()
    f1 = K_matrix[0, 2 ,2]
    f2 = K_matrix[0, 3, 2]
    sdepth = f1 + f2/depth
    xy_sdepth = torch.cat([xy_depth[...,:-1], sdepth.unsqueeze(-1)], dim=-1)

    to_ndc = None
    if world_coordinates:
        to_ndc = camera.get_full_projection_transform()
    else:
        to_ndc = camera.get_projection_transform()
    unproject_matrix = to_ndc.inverse().get_matrix()
    
    unprojected_points = transform_points_batch(xy_sdepth, unproject_matrix)
    return unprojected_points

def unproject_points(
    camera,
    ndc_points, #(..., 3)
    depth, #(..., )
):  
    '''
    unproject ndc_points into world coordinates
    '''
    k_inv = get_camera_k_inv(camera)
    proj = k_inv.new_zeros(4, 4)
    proj[:3, :3] = k_inv 
    proj[-1, -1] = 1
    trans = proj @ get_cam2world(camera)
    ndc_points *= depth.unsqueeze(-1)
    points_world = rotate_translate(ndc_points, trans)
    return points_world

def get_normalized_direction(
    camera,
    points
):  
    '''
    Args
        points: (..., 3) in world coordinate
    Return
        directions: (..., 3) 
    '''
    shape = points.size()
    points = points.view(-1, 3)
    camera_center = get_camera_center(camera)
    directions = points - camera_center 
    normalized = F.normalize(directions, dim=-1)
    normalized = normalized.view(*shape)
    return normalized

if __name__ == '__main__':
    test_intersect()
