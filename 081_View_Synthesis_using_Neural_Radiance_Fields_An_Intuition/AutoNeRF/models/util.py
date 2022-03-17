import matplotlib.pyplot as plt
import torch 
import numpy as np
import torchvision


def swap_channels(x):
    return x.permute(0, 3, 1, 2).contiguous()

def unswap_channels(x):
    return x.permute(0, 2, 3, 1).contiguous()


def show(image, permutation=(0,1,2)):
    grid_img = torchvision.utils.make_grid(image, nrow=5)
    plt.imshow(grid_img.permute(permutation))
    plt.axis('off')
    plt.show()



# def split_t(t):
#     if len(t.shape) == 1:
#         t=t.unsqueeze(dim=0)
#     print(t.shape)
#     tx = t[:, 0].detach().cpu().numpy()
#     ty = t[:, 1].detach().cpu().numpy()
#     tz = t[:, 2].detach().cpu().numpy()
#     return tx, ty, tz







# def get_cam_direction(poses):
#     #print(len(poses.shape), "sssssssssss")
#     if len(poses.shape) == 2:
#         poses = poses.unsqueeze(dim=0)
#     directions = torch.zeros((poses.shape[0], 3)).to(device)

#     for i in range(0, poses.shape[0]):

#         ori_, dir_ = nerf.get_ray_bundle(1,1, focal_length, poses[i])
#         dir_ = dir_.squeeze(dim=0).squeeze(dim=0)
#         directions[i] = dir_ + ori_

#     return directions



# def split_ray_points(dir, trans):
#     ray_lines_x = []
#     ray_lines_y = []
#     ray_lines_z = []


#     for i in range(0, dir.shape[0]):
#         ray_lines_x.append(dir[i][0].item())
#         ray_lines_x.append(trans[i][0].item())
#         ray_lines_x.append(None)

#         ray_lines_y.append(dir[i][1].item())
#         ray_lines_y.append(trans[i][1].item())
#         ray_lines_y.append(None)

#         ray_lines_z.append(dir[i][2].item())
#         ray_lines_z.append(trans[i][2].item())
#         ray_lines_z.append(None)
        
#     return ray_lines_x, ray_lines_y, ray_lines_z


# def plot_all_poses(translation, tform_cam2world):
#     x, y, z = split_t(translation)
#     xc, yc, zc = split_t(center)
#     xr, yr, zr = split_ray_points(get_cam_direction(tform_cam2world), translation)
#     xd, yd, zd = split_t(get_cam_direction(tform_cam2world))



#     new_pose = get_new_pose(center, radius).to(device)
#     print(new_pose)
#     new_t = new_pose[:3,3].unsqueeze(dim=0)
#     xn, yn, zn = split_t(new_t)
#     new_d = get_cam_direction(new_pose)
#     a, b, c = split_ray_points(new_d, new_t)


#     data=[go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=2.1), opacity=1, name="camera position"),
#           go.Scatter3d(x=xc, y=xc, z=xc,mode='markers', marker=dict(size=2.1), opacity=1, name="center"),
#           go.Scatter3d(x=xr, y=yr, z=zr,mode="lines", marker=dict(size=2.1), name="rays"),
#           go.Scatter3d(x=xd, y=yd, z=zd,mode='markers', marker=dict(size=2.1), name="rays"),

#           go.Scatter3d(x=xn, y=yn, z=zn,mode='markers', marker=dict(size=2.1), name="new translation"),
#           go.Scatter3d(x=a, y=b, z=c,mode='lines', marker=dict(size=2.1), name="new translation")
#     ]


#     fig = go.Figure(data)
#     fig.update_layout(title="Camera poses from the dataset", template = "plotly_dark")
#     fig.show()