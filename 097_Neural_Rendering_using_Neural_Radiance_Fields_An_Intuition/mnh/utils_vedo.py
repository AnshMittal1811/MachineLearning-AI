import numpy as np 
import torch 
import vedo 
from .plane_geometry import orthonormal_basis_from_xy

def get_vedo_cameras(
    R,
    T,
    arrow_len=1,
    s=1
):  
    '''
    get vedo object representing cameras 
    R, T: world2camera transform
    x: green, y: blue, z: red
    '''
    rotations = R
    positions = torch.bmm(R, -T.unsqueeze(-1)).squeeze(-1)
    x_end = positions + rotations[:,:,0] * arrow_len
    y_end = positions + rotations[:,:,1] * arrow_len
    z_end = positions + rotations[:,:,2] * arrow_len

    x = vedo.Arrows(positions, x_end, s=s, c='green')
    y = vedo.Arrows(positions, y_end, s=s, c='blue')
    z = vedo.Arrows(positions, z_end, s=s, c='red')
    return (x, y, z)

def get_vedo_cameras_cones(
    R, 
    T,
    r:float,
    height:float, 
    color,
    alpha:float=0.5
):
    axes = - R[:,:,-1]
    positions = torch.bmm(R, -T.unsqueeze(-1)).squeeze(-1)
    
    cones = []
    cam_num = R.size(0)
    for i in range(cam_num):
        cone = vedo.Cone(
            pos=list(positions[i]),
            axis=list(axes[i]),
            r=r,
            height=height,
            c=color,
            alpha=alpha
        )
        cones.append(cone)
    return cones

def get_vedo_alpha_plane(
    center, #(3,)
    rotation, #(3, 3)
    wh, #(2, )
    alpha, #(res_h, res_w)
    color=(0.5, 0.5, 0.5)
):  
    '''
    '''
    res_h, res_w = alpha.shape 
    w, h = wh
    vec_x, vec_y = rotation[:,0], rotation[:,1]
    verts = []
    faces = []
    for i_h in range(res_h):
        for i_w in range(res_w):
            len_x0 = (1/2 - i_w/res_w) * w  
            len_x1 = (1/2 - (i_w+1)/res_w) * w
            len_y0 = (1/2 - i_h/res_h) * h  
            len_y1 = (1/2 - (i_h+1)/res_h) * h
            v_0 = center + len_x0 * vec_x + len_y0 * vec_y
            v_1 = center + len_x1 * vec_x + len_y0 * vec_y
            v_2 = center + len_x1 * vec_x + len_y1 * vec_y
            v_3 = center + len_x0 * vec_x + len_y1 * vec_y
            id_base = len(verts)
            id_0 = id_base + 0
            id_1 = id_base + 1
            id_2 = id_base + 2
            id_3 = id_base + 3
            faces += [[id_0, id_1, id_2], [id_2, id_3, id_0]]
            verts += [v_0, v_1, v_2, v_3]
            
    alpha = np.stack([alpha, alpha], axis=-1)
    alpha = alpha.reshape(-1)
    color = [color for i in range(res_h*res_w*2)]
    plane = vedo.Mesh([verts, faces])
    plane.cellIndividualColors(color, alpha, alphaPerCell=True)
    return plane

def visualize_geometry(
    points,
    model,
    r:float=2,
    c:list=(0.5,0.5,0.5),
    alpha:float=0.5,
    screenshot_name:str=None
):
    objs = []
    points = points.cpu().numpy() 
    points = vedo.Points(points, r=r, c=c, alpha=1)
    objs.append(points)
    
    center = model.center.detach().cpu().numpy()
    xyz = orthonormal_basis_from_xy(model.xy.detach()).detach().cpu().numpy()
    wh = model.wh.detach().cpu().numpy()

    colors = np.random.rand(model.n_plane, 3)
    for i in range(model.n_plane):
        c = center[i]
        x, y = xyz[i,:,0], xyz[i,:,1]
        x_s, y_s = x*(wh[i, 0]/2), y*(wh[i, 1]/2)
        verts = [c+x_s+y_s, c-x_s+y_s, c-x_s-y_s, c+x_s-y_s]
        faces = [[0,1,2], [2,3,0]]
        plane = vedo.Mesh([verts, faces], c=colors[i], alpha=alpha)
        objs.append(plane)
        
    vedo.show(*objs ,axes=1)
    if screenshot_name:
        vedo.io.screenshot(screenshot_name)