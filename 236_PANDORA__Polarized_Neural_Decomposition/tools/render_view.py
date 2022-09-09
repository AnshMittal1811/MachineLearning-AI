from utils import io_util, mesh_util, rend_util
from models.frameworks import get_model
from utils.checkpoints import sorted_ckpts
from utils.print_fn import log

from src.polarization import cues_from_stokes, stokes_from_normal_rad, colorize_cues
from src.utils import linear_rgb_to_srgb_np, spec_srgb_lin, linear_rgb_to_srgb, srgb_to_linear_rgb_np

import os
import math
import imageio
import functools
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from PIL import Image

from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

def colorize_cues_np(pol_cues, color_aolp=False):
    # Convert polarization cues to color 
    # aolp
    aolp_sel = pol_cues['aolp'][...,0]/180. # Select the red channel normalize to 1
    # aolp_cmap = plt.get_cmap('hsv')
    aolp_cmap = plt.get_cmap('twilight')
    aolp_color = aolp_cmap(aolp_sel)[:,:,:3]
    # dop
    dop_sel = pol_cues['dop'][...,0] # Select the red channel
    dop_cmap = plt.get_cmap('viridis')
    dop_color = dop_cmap(dop_sel)[:,:,:3]

    if color_aolp:
        import polanalyser as pa
        import cv2
        aolp_dop_color = \
            cv2.cvtColor(
                pa.applyColorToAoLP(aolp_sel*np.pi,
                                    value=dop_sel), 
                cv2.COLOR_BGR2RGB)/255.
    else:
        aolp_dop_color = None
    return aolp_color, dop_color, aolp_dop_color
def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)

def view_matrix(
    forward: np.ndarray, 
    up: np.ndarray,
    cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    forward = poses[:, :3, 2].sum(0)
    up = poses[:, :3, 1].sum(0)
    c2w = view_matrix(forward, up, center)
    return c2w

def look_at(
    cam_location: np.ndarray, 
    point: np.ndarray, 
    up=np.array([0., -1., 0.])          # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
    ):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)     # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)

def c2w_track_spiral(c2w, up_vec, rads, focus: float, zrate: float, rots: int, N: int, zdelta: float = 0.):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change 
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.])
    
    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.])      # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4], 
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads        # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks


def smoothed_motion_interpolation(full_range, num_samples, uniform_proportion=1/3.):
    half_acc_proportion = (1-uniform_proportion) / 2.
    num_uniform_acc = max(math.ceil(num_samples*half_acc_proportion), 2)
    num_uniform = max(math.ceil(num_samples*uniform_proportion), 2)
    num_samples = num_uniform_acc * 2 + num_uniform
    seg_velocity = np.arange(num_uniform_acc)
    seg_angle = np.cumsum(seg_velocity)
    # NOTE: full angle = 2*k*x_max + k*v_max*num_uniform
    ratio = full_range / (2.0*seg_angle.max()+seg_velocity.max()*num_uniform)
    # uniform acceleration sequence
    seg_acc = seg_angle * ratio

    acc_angle = seg_acc.max()
    # uniform sequence
    seg_uniform = np.linspace(acc_angle, full_range-acc_angle, num_uniform+2)[1:-1]
    # full sequence
    all_samples = np.concatenate([seg_acc, seg_uniform, full_range-np.flip(seg_acc)])
    return all_samples


def visualize_cam_on_circle(intr, extrs, up_vec, c0):
    
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from tools.vis_camera import draw_camera
    
    cam_width = 0.2/2     # Width/2 of the displayed camera.
    cam_height = 0.1/2    # Height/2 of the displayed camera.
    scale_focal = 2000        # Value to scale the focal length.
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    ax.set_aspect("auto")    

    matplotlib.rcParams.update({'font.size': 22})
    #----------- draw cameras
    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrs, True)

    radius = np.linalg.norm(c0)
    
    #----------- draw small circle
    angles = np.linspace(0, np.pi * 2., 180)
    rots = R.from_rotvec(angles[:, None] * up_vec[None, :])
    # [180, 3]
    pts = rots.apply(c0)
    # [x, z, -y]
    ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], color='black')
    
    #----------- draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
    
    #----------- draw axis
    axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    X, Y, Z, U, V, W = zip(*axis) 
    ax.quiver(X[0], Z[0], -Y[0], U[0], W[0], -V[0], color='red')
    ax.quiver(X[1], Z[1], -Y[1], U[1], W[1], -V[1], color='green')
    ax.quiver(X[2], Z[2], -Y[2], U[2], W[2], -V[2], color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    
    plt.show()


def visualize_cam_spherical_spiral(intr, extrs, up_vec, c0, focus_center, n_rots, up_angle):
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    from tools.vis_camera import draw_camera
    
    cam_width = 0.2/2     # Width/2 of the displayed camera.
    cam_height = 0.1/2    # Height/2 of the displayed camera.
    scale_focal = 2000        # Value to scale the focal length.
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    ax.set_aspect("auto")    

    matplotlib.rcParams.update({'font.size': 22})
    #----------- draw cameras
    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrs, True)

    radius = np.linalg.norm(c0)
    
    #----------- draw small circle
    # key rotations of a spherical spiral path
    num_pts = int(n_rots * 180.)
    sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, num_pts)
    sphere_phis = np.linspace(0, up_angle, num_pts)
    # first rotate about up vec
    rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
    pts = rots_theta.apply(c0)
    # then rotate about horizontal vec
    horizontal_vec = normalize(np.cross(pts-focus_center[None, :], up_vec[None, :], axis=-1))
    rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
    pts = rots_phi.apply(pts)
    # [x, z, -y]
    ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], color='black')
    
    #----------- draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
    
    #----------- draw axis
    axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    X, Y, Z, U, V, W = zip(*axis) 
    ax.quiver(X[0], Z[0], -Y[0], U[0], W[0], -V[0], color='red')
    ax.quiver(X[1], Z[1], -Y[1], U[1], W[1], -V[1], color='green')
    ax.quiver(X[2], Z[2], -Y[2], U[2], W[2], -V[2], color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    
    plt.savefig('viz/spherical_spiral_path.png')

def integerify(img):
    return (np.minimum(img,1)*255.).astype(np.uint8)


def save_gt_images(dataset, args, c2ws, intrinsics, H, W, save_dir, gt_space,polarized):
    if gt_space == 'linear':
        to_space_np = lambda x:np.clip(linear_rgb_to_srgb_np(x),0,1)
    elif gt_space == 'srgb':
        to_space_np = lambda x:x

    view_ids = args.camera_inds.split(',')
    view_ids = [int(v) for v in view_ids]
    to_img_np = lambda img: img.reshape(H,W,3).cpu().numpy()
    to_mask_np = lambda img: img.reshape(H,W).cpu().numpy()
    for view_id in view_ids:
        c2w = c2ws[view_id]
        rays_o, rays_d, _ = rend_util.get_rays(torch.from_numpy(c2w).float()[None,...],
                                               intrinsics[None,...], H, W, N_rays=-1)
        # Similar code as in train.py
        if  not args.data.gt_type == 'stokes':
            target_rgb = dataset.rgb_images[view_id]

            imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_rgb.png'),
                        integerify(to_space_np(to_img_np(target_rgb))))

            if not args.model.only_diffuse:
                target_specular = dataset.specular_images[view_id]
                imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_specular.png'),
                            integerify(to_space_np(to_img_np(target_specular))))
        
        if polarized:
            if args.data.gt_type == 'normal':
                target_normal = dataset.normal_images[view_id]
                imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_normal.png'),
                                integerify(0.5+0.5*(to_img_np(target_normal))))
                if args.model.only_diffuse:
                    target_stokes = stokes_from_normal_rad(rays_o, rays_d, target_normal[None], 
                                    target_rgb[None], train_mode=True)
                else:
                    target_stokes = stokes_from_normal_rad(rays_o, rays_d, target_normal[None], 
                                    target_rgb[None], spec_rads=target_specular[None], 
                                    train_mode=True)
            elif args.data.gt_type == 'stokes':
                target_stokes = torch.stack([dataset.s0_images[view_id], 
                                             dataset.s1_images[view_id],
                                             dataset.s2_images[view_id]], -1)
            else:
                raise Exception(f'Invalid data gt_type {args.data.gt_type}. Options: stokes, normal')
            target_cues = colorize_cues(cues_from_stokes(target_stokes),
                                        gamma_s0=(gt_space=='linear'),
                                        color_aolp = args.color_aolp)
            for cue_name, cue_val in target_cues.items():
                imageio.imwrite(os.path.join(save_dir,f'gt_{view_id:03d}_{cue_name}.png'),
                                integerify(to_img_np(cue_val)))
        
        if len(dataset.object_masks)>0:
            target_mask = dataset.object_masks[view_id]
            imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_mask.png'),
                        integerify((to_mask_np(target_mask+0.))))

def write_env_map_tb(spec_net, savename, dev,
                     fres_in_mlp=False, use_roughness=True, 
                     roughness=0.01):

    npts_u = 1000
    npts_v = 500 
    nchannels = 3

    u_mat, v_mat = torch.meshgrid(torch.linspace(-math.pi, math.pi, npts_u), torch.linspace(0, math.pi, npts_v))
    dx = torch.reshape(torch.cos(u_mat) * torch.sin(v_mat), (1, -1, 1))
    dy = torch.reshape(torch.sin(u_mat) * torch.sin(v_mat), (1, -1, 1))
    dz = torch.reshape(torch.cos(v_mat), (1, -1, 1))

    coords = torch.cat((dx, dy, dz), dim=-1).to(dev)
    roughness_torch = roughness*torch.ones(dx.shape, dtype=dx.dtype, device=dev)

    if use_roughness:
        if not fres_in_mlp:
            env_map = torch.abs(spec_net.forward(coords,roughness_torch))
        else:
            cos_theta = torch.ones(dx.shape, device=dev, dtype=dx.dtype)
            env_map = spec_net.forward(coords, roughness_torch,cos_theta)/0.04
    else:
        if not fres_in_mlp:
            env_map = torch.abs(spec_net.forward(coords))
        else:
            cos_theta = torch.ones(dx.shape, device=dev, dtype=dx.dtype)
            env_map = spec_net.forward(coords, cos_theta)/0.04
    env_map_torch = torch.reshape(env_map, (npts_u, npts_v, nchannels))

    # env_map_torch = (spec_srgb_lin(env_map_torch, toLin=True)) # gamma parameter in spec_srgb_lin maybe different from srgb_to_linear_rgb

    env_map_torch = torch.flip(env_map_torch.permute((1,0,2)), dims=[0])

    env_map_tb = torch.clip(linear_rgb_to_srgb(env_map_torch).permute((2,0,1)),
                            min=0., max=1.)
    # env_map_tb = linear_rgb_to_srgb(env_map_torch).permute((2,0,1))
    # env_map_tb = env_map_tb * (1. / torch.amax(env_map_tb, dim=(0,1,2)))

    return env_map_tb


def write_env_map(spec_net, savename, dev,
                  fres_in_mlp = False, use_roughness=True, 
                  roughness=0.2):

    npts_u = 1000
    npts_v = 500 
    nchannels = 3

    u_mat, v_mat = torch.meshgrid(torch.linspace(-math.pi, math.pi, npts_u), torch.linspace(0, math.pi, npts_v))
    # u_mat, v_mat = torch.meshgrid(torch.linspace(0, 2*math.pi, npts_u), torch.linspace(0, math.pi, npts_v))
    dx = torch.reshape(torch.cos(u_mat) * torch.sin(v_mat), (1, -1, 1))
    dy = torch.reshape(torch.sin(u_mat) * torch.sin(v_mat), (1, -1, 1))
    dz = torch.reshape(torch.cos(v_mat), (1, -1, 1))

    coords = torch.cat((dx, dy, dz), dim=-1).to(dev)
    roughness_torch = roughness*torch.ones(dx.shape, dtype=dx.dtype, device=dev)

    if use_roughness:
        if not fres_in_mlp:
            env_map = torch.abs(spec_net.forward(coords,roughness_torch))
        else:
            cos_theta = torch.ones(dx.shape, device=dev, dtype=dx.dtype)
            env_map = spec_net.forward(coords, roughness_torch, cos_theta)/0.04
    else:
        if not fres_in_mlp:
            env_map = torch.abs(spec_net.forward(coords))
        else:
            cos_theta = torch.ones(dx.shape, device=dev, dtype=dx.dtype)
            env_map = spec_net.forward(coords,cos_theta)/0.04
    env_map_torch = torch.reshape(env_map, (npts_u, npts_v, nchannels))

    # env_map_torch = (spec_srgb_lin(env_map_torch, toLin=True)) # gamma parameter in spec_srgb_lin maybe different from srgb_to_linear_rgb

    env_map_torch = torch.flip(env_map_torch.permute((1,0,2)), dims=[0])

    imageio.imwrite(f"{savename}.exr", env_map_torch.cpu().detach().numpy())


    for i in range(3):
        env_map_torch /= 2**i
        env_map_png = linear_rgb_to_srgb(env_map_torch)

        # env_map_png = env_map_png * (255 / torch.amax(env_map_png, dim=(0,1,2)))
        env_map_png = torch.clip(env_map_png,min=0.,max=1.) * 255
        env_map_png = env_map_png.to(torch.uint8)
        imageio.imwrite(f"{savename}_pow2=-{i}.png", env_map_png.cpu().detach().numpy())
    print("Wrote image to %s" % (savename))

    return env_map_png.permute((2,0,1))

def calculate_psnr(img1, img2, mask):
    # return psnr(img1,img2)
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def mySaveIm(fname, im_arr):
    from PIL import Image
    from matplotlib import cm
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

    im_arr = np.nan_to_num(im_arr,copy=True,nan=0.0,posinf=0.0,neginf=0.0)
    orig_im = im_arr 
    
    im_arr = cm.viridis(np.squeeze(im_arr/np.amax(im_arr)))
    im_arr = np.uint8(im_arr * 255.)
    saveim = Image.fromarray(im_arr)
    saveim.save("%s.png" % fname)

    plt.figure()
    mpb = plt.imshow(orig_im)
    fig,ax = plt.subplots()
    plt.colorbar(mpb, ax=ax)
    ax.remove()
    # plt.show()
    plt.savefig("%s_cmap.png" % fname, bbox_inches='tight', transparent=True)
    plt.close()

def main_function(args):
    do_render_mesh = args.render_mesh is not None
    if do_render_mesh:
        import open3d as o3d
    
    io_util.cond_mkdir('./out')

    # Override polarization flag for polarized rendering
    orig_pol = args.model.polarized
    args.model.polarized = args.render_pol
    args.model.disable_fres = args.material_edit == "metallic"
    model, trainer, render_kwargs_train, render_kwargs_test, render_fn = get_model(args)
    if args.load_pt is None:
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.training.exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.load_pt
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=args.device)
    model.load_state_dict(state_dict['model'])
    model.to(args.device)
    
    calc_rough = not(args.model.use_env_mlp in ['no_envmap_MLP', 'mask_no_envmap_MLP'])

    if not (args.model.use_env_mlp in ['no_envmap_MLP', 'mask_no_envmap_MLP']):
    # if args.gen_env_map:
        outbase = args.expname if args.outbase is None else args.outbase
        save_dir = os.path.join('out',f"{outbase}_imgs")
        os.makedirs(save_dir, exist_ok=True )
        alpha_L = args.alpha_vals.split(',')
        alpha_L = [float(a) for a in alpha_L]
        for i in range(len(alpha_L)):
            alpha_str = str(alpha_L[i])
            alpha_str.replace('.','p')
            write_env_map(model.specular_net, f'{save_dir}/env_map_alpha={alpha_str}', args.device,
                roughness=alpha_L[i])
        # write_env_map(model.specular_net, f'{save_dir}/est_env_map', args.device,
        #             fres_in_mlp=args.model.env_mlp_type == 'fres_input',
        #             use_roughness=args.model.use_env_mlp == 'rough_envmap_MLP')

        # # return

    if args.use_surface_render:
        assert args.use_surface_render == 'sphere_tracing' or args.use_surface_render == 'root_finding'
        from models.ray_casting import surface_render
        render_fn = functools.partial(surface_render, model=model, ray_casting_algo=args.use_surface_render, diffuse_only=args.diffuse_only)
    
    if args.alter_radiance is not None:
        state_dict = torch.load(args.alter_radiance, map_location=args.device)
        radiance_state_dict = {}
        for k, v in state_dict['model'].items():
            if 'radiance_net' in k:
                newk = k.replace('radiance_net.', '')
                radiance_state_dict[newk] = v
        model.radiance_net.load_state_dict(radiance_state_dict)

    from dataio import get_data
    dataset = get_data(args, downscale=args.downscale)

    (_, model_input, ground_truth) = dataset[0]
    intrinsics = model_input["intrinsics"].cuda()
    H, W = (dataset.H, dataset.W)
    # NOTE: fx, fy should be scalec with the same ratio. Different ratio will cause the picture itself be stretched.
    #       fx=intrinsics[0,0]                   fy=intrinsics[1,1]
    #       cy=intrinsics[1,2] for H's scal      cx=intrinsics[0,2] for W's scale
    if args.H is not None:
        intrinsics[1,2] *= (args.H/dataset.H)
        H = args.H
    if args.H_scale is not None:
        H = int(dataset.H * args.H_scale)
        intrinsics[1,2] *= (H/dataset.H)

    if args.W is not None:
        intrinsics[0,2] *= (args.W/dataset.W)
        W = args.W
    if args.W_scale is not None:
        W = int(dataset.W * args.W_scale)
        intrinsics[0,2] *= (W/dataset.W)
    log.info("=> Rendering resolution @ [{} x {}]".format(H, W))

    c2ws = torch.stack(dataset.c2w_all, dim=0).data.cpu().numpy()

    if args.data.space == 'linear':
        to_space_np = lambda x:np.clip(linear_rgb_to_srgb_np(x),0,1)
    elif args.data.space == 'srgb':
        to_space_np = lambda x:x


    # Save gt images for comparison
    if args.save_gt:
        outbase = args.expname if args.outbase is None else args.outbase
        post_fix = '{}x{}_{}_{}'.format(H, W, args.num_views, args.camera_path)
        save_dir = os.path.join('out',f"{outbase}_imgs")
        os.makedirs(save_dir, exist_ok=True )
        save_gt_images(dataset, args, c2ws, intrinsics, H, W, save_dir, args.data.space, orig_pol)

    #-----------------
    # Spiral path
    #   original nerf-like spiral path
    #-----------------
    if args.camera_path == 'spiral':
        c2w_center = poses_avg(c2ws)
        up = c2ws[:, :3, 1].sum(0)
        rads = np.percentile(np.abs(c2ws[:, :3, 3]), 30, 0)
        focus_distance = np.mean(np.linalg.norm(c2ws[:, :3, 3], axis=-1))
        render_c2ws = c2w_track_spiral(c2w_center, up, rads, focus_distance*0.8, zrate=0.0, rots=1, N=args.num_views)
    #-----------------
    # https://en.wikipedia.org/wiki/Spiral#Spherical_spirals
    #   assume three input views are on a small circle, then generate a spherical spiral path based on the small circle
    #-----------------
    elif args.camera_path == 'spherical_spiral':
        up_angle = np.pi / 3.
        n_rots = 2.2
        
        view_ids = args.camera_inds.split(',')
        assert len(view_ids) == 3, 'please select three views on a small circle, in CCW order (from above)'
        view_ids = [int(v) for v in view_ids]
        centers = c2ws[view_ids, :3, 3]
        centers_norm = np.linalg.norm(centers, axis=-1)
        radius = np.max(centers_norm)
        centers = centers * radius / centers_norm
        vec0 = centers[1] - centers[0]
        vec1 = centers[2] - centers[0]
        # the axis vertical to the small circle's area
        up_vec = normalize(np.cross(vec0, vec1))
                
        # key rotations of a spherical spiral path
        sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, args.num_views)
        sphere_phis = np.linspace(0, up_angle, args.num_views)
        
        if True:
            # use the origin as the focus center
            focus_center = np.zeros([3])
        else:
            # use the center of the small circle as the focus center
            focus_center = np.dot(up_vec, centers[0]) * up_vec
        
        # first rotate about up vec
        rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
        render_centers = rots_theta.apply(centers[0])
        # then rotate about horizontal vec
        horizontal_vec = normalize(np.cross(render_centers-focus_center[None, :], up_vec[None, :], axis=-1))
        rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
        render_centers = rots_phi.apply(render_centers)
        
        render_c2ws = look_at(render_centers, focus_center[None, :], up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_spherical_spiral(intr, extrs, up_vec, centers[0], focus_center, n_rots, up_angle)
            
    #------------------
    # Small Circle Path: 
    #   assume three input views are on a small circle, then interpolate along this small circle
    #------------------
    elif args.camera_path == 'small_circle':
        view_ids = args.camera_inds.split(',')
        assert len(view_ids) == 3, 'please select three views on a small circle, int CCW order (from above)'
        view_ids = [int(v) for v in view_ids]
        centers = c2ws[view_ids, :3, 3]
        centers_norm = np.linalg.norm(centers, axis=-1)
        radius = np.max(centers_norm)
        centers = centers * radius / centers_norm
        vec0 = centers[1] - centers[0]
        vec1 = centers[2] - centers[0]
        # the axis vertical to the small circle
        up_vec = normalize(np.cross(vec0, vec1))
        # length of the chord between c0 and c2
        len_chord = np.linalg.norm(vec1, axis=-1)
        # angle of the smaller arc between c0 and c1
        full_angle = np.arcsin(len_chord/2/radius) * 2.
        
        all_angles = smoothed_motion_interpolation(full_angle, args.num_views)
        
        rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
        centers = rots.apply(centers[0])
        
        # get c2w matrices
        render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_on_circle(intr, extrs, up_vec, centers[0])
    #-----------------
    # Interpolate path
    #   directly interpolate among all input views
    #-----------------
    elif args.camera_path == 'interpolation':
        # c2ws = c2ws[:25]  # NOTE: [:20] fox taxi dataset
        key_rots = R.from_matrix(c2ws[:, :3, :3])
        key_times = list(range(len(key_rots)))
        slerp = Slerp(key_times, key_rots)
        interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
        render_c2ws = []
        for i in range(args.num_views):
            time = float(i) / args.num_views * (len(c2ws) - 1)
            cam_location = interp(time)
            cam_rot = slerp(time).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = cam_rot
            c2w[:3, 3] = cam_location
            render_c2ws.append(c2w)
        render_c2ws = np.stack(render_c2ws, axis=0)
    #------------------
    # Great Circle Path: 
    #   assume two input views are on a great circle, then interpolate along this great circle
    #------------------
    elif args.camera_path == 'great_circle':
        # to interpolate along a great circle that pass through the c2w center of view0 and view1
        view01 = args.camera_inds.split(',')
        assert len(view01) == 2, 'please select two views on a great circle, in CCW order (from above)'
        view0, view1 = [int(s) for s in view01]
        c0 = c2ws[view0, :3, 3]
        c0_norm = np.linalg.norm(c0)
        c1 = c2ws[view1, :3, 3]
        c1_norm = np.linalg.norm(c1)
        # the radius of the great circle
        # radius = (c0_norm+c1_norm)/2.
        radius = max(c0_norm, c1_norm)
        # re-normalize the c2w centers to be on the exact same great circle
        c0 = c0 * radius / c0_norm
        c1 = c1 * radius / c1_norm
        # the axis vertical to the great circle
        up_vec = normalize(np.cross(c0, c1))
        # length of the chord between c0 and c1
        len_chord = np.linalg.norm(c0-c1, axis=-1)
        # angle of the smaller arc between c0 and c1
        full_angle = np.arcsin(len_chord/2/radius) * 2.
        
        all_angles = smoothed_motion_interpolation(full_angle, args.num_views)
        
        # get camera centers
        rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
        centers = rots.apply(c0)
        
        # get c2w matrices
        render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_on_circle(intr, extrs, up_vec, centers[0])
    
    elif args.camera_path == 'train_views':
        view_ids = args.camera_inds.split(',')
        view_ids = [int(v) for v in view_ids]
        render_c2ws = c2ws[view_ids]
    
    elif args.camera_path == 'all_train_views':
        view_ids = range(c2ws.shape[0])
        render_c2ws = c2ws

    else:
        raise RuntimeError("Please choose render type between [spiral, interpolation, small_circle, great_circle, spherical_spiral,train_views, all_train_views]")
    log.info("=> Camera path: {}".format(args.camera_path))



    # if args.extract_mesh:
    #     with torch.no_grad():
    #         savestr = "out/%s_imgs/%s_mesh.ply" % (outbase, outbase)
    #         mesh_util.extract_mesh(
    #             model.implicit_surface, 
    #             filepath=savestr,
    #             volume_size=args.data.get('volume_size', 1.5),
    #             show_progress=True,
    #             N=1024)




    rgb_imgs = []
    depth_imgs = []
    normal_imgs = []
    normal_imgs_orig = []
    alb_imgs = []
    # save mesh render images
    mesh_imgs = []
    render_kwargs_test['rayschunk'] = args.rayschunk

    if args.model.polarized:
        s0_imgs = []
        dop_imgs = []
        aolp_imgs = []
        if args.color_aolp:
            color_aolp_imgs = []
        if not args.model.only_diffuse:
            diff_s0_imgs = []
            diff_dop_imgs = []
            diff_aolp_imgs = []
            spec_s0_imgs = []
            spec_dop_imgs = []
            spec_aolp_imgs = []

    if not args.model.only_diffuse:
        spec_imgs = []
        spec_fac0_imgs = []
        if calc_rough:
            rough_imgs = []

    if 'mask' in args.model.use_env_mlp:
        mask_imgs = []

    if do_render_mesh:
        log.info("=> Load mesh: {}".format(args.render_mesh))
        geometry = o3d.io.read_triangle_mesh(args.render_mesh)
        geometry.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=W, height=H, visible=args.debug)
        ctrl = vis.get_view_control()
        vis.add_geometry(geometry)
        # opt = vis.get_render_option()
        # opt.mesh_show_back_face = True
        
        cam = ctrl.convert_to_pinhole_camera_parameters()
        intr = intrinsics.data.cpu().numpy()
        # cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], intr[0,2], intr[1,2])
        cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], W/2-0.5, H/2-0.5)
        ctrl.convert_from_pinhole_camera_parameters(cam)
        

        
    for c2w in tqdm(render_c2ws, desc='rendering...') :
        if not args.debug and not args.disable_rgb:
            rays_o, rays_d, select_inds = rend_util.get_rays(torch.from_numpy(c2w).float().cuda()[None, ...], intrinsics[None, ...], H, W, N_rays=-1)
            with torch.no_grad():
                # NOTE: detailed_output set to False to save a lot of GPU memory.
                render_kwargs_test["relight_map"] = args.relight_map
                rgb, depth, extras = render_fn(
                    rays_o, rays_d, show_progress=True, calc_normal=True, detailed_output=False, **render_kwargs_test)
                depth = depth.data.cpu().reshape(H, W, 1).numpy()
                depth = depth/depth.max()
                rgb_imgs.append(rgb.data.cpu().reshape(H, W, 3).numpy())
                depth_imgs.append(depth)
                if args.use_surface_render:
                    normals = extras['normals_surface']
                else:
                    normals = extras['normals_volume']
                normals = normals.data.cpu().reshape(H, W, 3).numpy()
                # if True:
                #     # (c2w^(-1) @ n)^T = n^T @ c2w^(-1)^T = n^T @ c2w
                #     normals = normals @ c2w[:3, :3]
                normal_imgs.append(normals/2.+0.5)
                normal_imgs_orig.append(normals)

                alb_imgs.append(extras['albedo'].data.cpu().reshape(H,W,3).numpy())
                
                if args.model.polarized:
                    # Get polarization cues
                    s0 = extras['s0'].data.cpu().reshape(H,W,3).numpy()
                    s1 = extras['s1'].data.cpu().reshape(H,W,3).numpy()
                    s2 = extras['s2'].data.cpu().reshape(H,W,3).numpy()
                    stokes_stack = np.stack([s0,s1,s2],-1)
                    pol_cues = cues_from_stokes(stokes_stack)

                    aolp_color, dop_color, aolp_dop_color = colorize_cues_np(pol_cues, 
                                                                             args.color_aolp)
                    aolp_imgs.append(aolp_color)
                    dop_imgs.append(dop_color)
                    s0_imgs.append(pol_cues['s0'])
                    if args.color_aolp:
                        color_aolp_imgs.append(aolp_dop_color)
                    
                    if not args.model.only_diffuse:
                        # Get polarization cues
                        diff_s0 = extras['diff_s0'].data.cpu().reshape(H,W,3).numpy()
                        diff_s1 = extras['diff_s1'].data.cpu().reshape(H,W,3).numpy()
                        diff_s2 = extras['diff_s2'].data.cpu().reshape(H,W,3).numpy()
                        diff_stokes_stack = np.stack([diff_s0,
                                                      diff_s1,
                                                      diff_s2],-1)
                        diff_pol_cues = cues_from_stokes(diff_stokes_stack)

                        diff_aolp_color, diff_dop_color, diff_aolp_dop_color = colorize_cues_np(diff_pol_cues)
                        diff_aolp_imgs.append(diff_aolp_color)
                        diff_dop_imgs.append(diff_dop_color)
                        diff_s0_imgs.append(diff_pol_cues['s0'])

                        spec_s0 = extras['spec_s0'].data.cpu().reshape(H,W,3).numpy()
                        spec_s1 = extras['spec_s1'].data.cpu().reshape(H,W,3).numpy()
                        spec_s2 = extras['spec_s2'].data.cpu().reshape(H,W,3).numpy()
                        spec_stokes_stack = np.stack([spec_s0,
                                                      spec_s1,
                                                      spec_s2],-1)
                        spec_pol_cues = cues_from_stokes(spec_stokes_stack)

                        spec_aolp_color, spec_dop_color, spec_aolp_dop_color = colorize_cues_np(spec_pol_cues)
                        spec_aolp_imgs.append(spec_aolp_color)
                        spec_dop_imgs.append(spec_dop_color)
                        spec_s0_imgs.append(spec_pol_cues['s0'])
                    
                if not args.model.only_diffuse:
                    spec = extras['spec_map']
                    spec_imgs.append(spec.data.cpu().reshape(H,W,3).numpy())
                    if 'spec_fac0' in extras:
                        spec_fac0 = extras['spec_fac0']
                        spec_fac0_imgs.append(spec_fac0.data.cpu().reshape(H,W).numpy())
                    if calc_rough:
                        rough_imgs.append(extras['rough_map'].cpu().reshape(H,W,1).numpy())
                if 'mask' in args.model.use_env_mlp:
                    mask = extras['mask_map']
                    mask_imgs.append(mask.data.cpu().reshape(H,W).numpy())
        if do_render_mesh:
            extr = np.linalg.inv(c2w)
            cam.extrinsic = extr
            ctrl.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            if not args.debug:
                rgb_mesh = vis.capture_screen_float_buffer(do_render=True)
                mesh_imgs.append(np.asarray(rgb_mesh))

    def integerify(img):
        return (img*255.).astype(np.uint8)

    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_alb.exr'), alb) 
        for idx,alb in enumerate(alb_imgs)]

    if args.material_edit == 'swap_rg':
        # Swap r and g channels
        mat_edit_imgs = [integerify(to_space_np(rgb_img[...,[1,0,2]]+spec_img)) 
                            for rgb_img,spec_img in zip(diff_s0_imgs,spec_s0_imgs)]
    elif args.material_edit == 'swap_rb':
        # Swap r and b channels
        mat_edit_imgs = [integerify(to_space_np(rgb_img[...,[2,1,0]]+spec_img)) 
                            for rgb_img,spec_img in zip(diff_s0_imgs,spec_s0_imgs)]
    elif args.material_edit == 'only_b':
        only_b = lambda x : np.stack([1.2*x[...,0],0.*x[...,0],x[...,2]],-1)
        mat_edit_imgs = [integerify(to_space_np(only_b(rgb_img)+spec_img)) 
                            for rgb_img,spec_img in zip(diff_s0_imgs,spec_s0_imgs)]
    mix_rgb_imgs = [integerify(to_space_np(rgb_img+spec_img)) 
                        for rgb_img,spec_img in zip(rgb_imgs,spec_imgs)]

    rgb_imgs = [integerify(to_space_np(img)) for img in rgb_imgs]
    depth_imgs = [integerify(img) for img in depth_imgs]
    normal_imgs = [integerify(img) for img in normal_imgs]
    mesh_imgs = [integerify(img) for img in mesh_imgs]
    alb_imgs = [integerify(to_space_np(img)) for img in alb_imgs]
    if args.model.polarized:
        dop_imgs = [integerify(img) for img in dop_imgs]
        aolp_imgs = [integerify(img) for img in aolp_imgs]
        s0_imgs = [integerify(to_space_np(img)) for img in s0_imgs]
        if args.color_aolp:
            color_aolp_imgs = [integerify(img) for img in color_aolp_imgs]
        if not args.model.only_diffuse:
            diff_dop_imgs = [integerify(img) for img in diff_dop_imgs]
            diff_aolp_imgs = [integerify(img) for img in diff_aolp_imgs]
            diff_s0_imgs = [integerify(to_space_np(img)) for img in diff_s0_imgs]
            spec_dop_imgs = [integerify(img) for img in spec_dop_imgs]
            spec_aolp_imgs = [integerify(img) for img in spec_aolp_imgs]
            spec_s0_imgs = [integerify(to_space_np(img)) for img in spec_s0_imgs]
    if not args.model.only_diffuse:
        spec_imgs = [integerify(to_space_np(img)) for img in spec_imgs]
        spec_fac0_imgs = [integerify((img)) for img in spec_fac0_imgs]

        if calc_rough:
            [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_rough.exr'), rough) 
                for idx,rough in enumerate(rough_imgs)]
            rough_imgs = [integerify((img)) for img in rough_imgs]

    if 'mask' in args.model.use_env_mlp:
        mask_imgs  = [(img>0.5) for img in mask_imgs]

    if args.mask_obj :
        for i in range(len(rgb_imgs)):
            if 'mask' in args.model.use_env_mlp:
                mask_i = mask_imgs[i][...,None]
            elif args.camera_path == 'train_views':
                mask_i = dataset.object_masks[view_ids[i]].reshape(H,W,1).cpu().detach().numpy()
            else:
                raise Exception('Cannot obtain masks')
            # For white background
            mask_fn = lambda x: (x*mask_i + (1-mask_i)*255.).astype('uint8')

            rgb_imgs[i] = mask_fn(rgb_imgs[i]) 
            mix_rgb_imgs[i] = mask_fn(mix_rgb_imgs[i])
            depth_imgs[i] = mask_fn(depth_imgs[i])
            normal_imgs[i] = mask_fn(normal_imgs[i])
            alb_imgs[i] = mask_fn(alb_imgs[i])
            if not args.model.only_diffuse:
                spec_imgs[i] = mask_fn(spec_imgs[i])
            if args.model.polarized:
                s0_imgs[i] = mask_fn(s0_imgs[i])
                dop_imgs[i] = mask_fn(dop_imgs[i])
                aolp_imgs[i] = mask_fn(aolp_imgs[i])
                if args.color_aolp:
                    color_aolp_imgs = mask_fn(color_aolp_imgs)
                if not args.model.only_diffuse:
                    diff_s0_imgs[i] = mask_fn(diff_s0_imgs[i])
                    diff_dop_imgs[i] = mask_fn(diff_dop_imgs[i])
                    diff_aolp_imgs[i] = mask_fn(diff_aolp_imgs[i])
                    spec_s0_imgs[i] = mask_fn(spec_s0_imgs[i])
                    spec_dop_imgs[i] = mask_fn(spec_dop_imgs[i])
                    spec_aolp_imgs[i] = mask_fn(spec_aolp_imgs[i])
                if args.material_edit in ['swap_rg','swap_bg','only_b']:
                    mat_edit_imgs[i] = mask_fn(mat_edit_imgs[i])

    if args.save_videos:
        if not args.debug:
            if args.outbase is None:
                outbase = args.expname
            else:
                outbase = args.outbase
            post_fix = '{}x{}_{}_{}'.format(H, W, args.num_views, args.camera_path)
            if args.use_surface_render:
                post_fix = post_fix + '_{}'.format(args.use_surface_render)
            save_dir = os.path.join('out',f"{outbase}_vids")
            os.makedirs(save_dir, exist_ok=True)
            if not args.disable_rgb:
                imageio.mimwrite(os.path.join(save_dir, '{}_rgb_{}.mp4'.format(outbase, post_fix)), rgb_imgs, fps=args.fps, quality=8)
                imageio.mimwrite(os.path.join(save_dir, '{}_depth_{}.mp4'.format(outbase, post_fix)), depth_imgs, fps=args.fps, quality=8)
                imageio.mimwrite(os.path.join(save_dir, '{}_normal_{}.mp4'.format(outbase, post_fix)), normal_imgs, fps=args.fps, quality=8)
                imageio.mimwrite(os.path.join(save_dir, '{}_albedo_{}.mp4'.format(outbase, post_fix)), alb_imgs, fps=args.fps, quality=8)
                if not args.model.only_diffuse:
                    imageio.mimwrite(os.path.join(save_dir, '{}_spec_{}.mp4'.format(outbase, post_fix)), spec_imgs, fps=args.fps, quality=8)
                    imageio.mimwrite(os.path.join(save_dir, '{}_mix_rgb_{}.mp4'.format(outbase, post_fix)), mix_rgb_imgs, fps=args.fps, quality=8)
                    if calc_rough:
                        imageio.mimwrite(os.path.join(save_dir, '{}_rough_{}.mp4'.format(outbase, post_fix)), rough_imgs, fps=args.fps, quality=8)
                #         rgb_and_spec_and_normal_imgs = [np.concatenate([rgb, spec, normal], axis=0) for rgb, spec, normal, alb, rough in zip(rgb_imgs, spec_imgs, normal_imgs, alb_imgs, rough_imgs)]
                #     else:
                #         rgb_and_spec_and_normal_imgs = [np.concatenate([rgb, spec, normal], axis=0) for rgb, spec, normal, alb in zip(rgb_imgs, spec_imgs, normal_imgs, alb_imgs)]
                #     imageio.mimwrite(os.path.join('out', '{}_rgb&spec&normal_{}.mp4'.format(outbase, post_fix)), rgb_and_spec_and_normal_imgs, fps=args.fps, quality=8)
                # else:
                #     rgb_and_normal_imgs = [np.concatenate([rgb, normal], axis=0) for rgb, normal in zip(rgb_imgs, normal_imgs)]
                #     imageio.mimwrite(os.path.join('out', '{}_rgb&normal_{}.mp4'.format(outbase, post_fix)), rgb_and_normal_imgs, fps=args.fps, quality=8)
            if do_render_mesh:
                vis.destroy_window()
                imageio.mimwrite(os.path.join(save_dir, '{}_mesh_{}.mp4'.format(outbase, post_fix)), mesh_imgs, fps=args.fps, quality=8)
                if not args.disable_rgb:
                    rgb_and_mesh_imgs = [np.concatenate([rgb, mesh], axis=0) for rgb, mesh in zip(rgb_imgs, mesh_imgs)]
                    imageio.mimwrite(os.path.join(save_dir, '{}_rgb&mesh_{}.mp4'.format(outbase, post_fix)), rgb_and_mesh_imgs, fps=args.fps, quality=8)
                    rgb_and_normal_and_mesh_imgs = [np.concatenate([rgb, normal, mesh], axis=0) for rgb, normal, mesh in zip(rgb_imgs, normal_imgs, mesh_imgs)]
                    imageio.mimwrite(os.path.join(save_dir, '{}_rgb&normal&mesh_{}.mp4'.format(outbase, post_fix)), rgb_and_normal_and_mesh_imgs, fps=args.fps, quality=8)
                    
            if args.model.polarized:
                pol_cues_imgs = [np.concatenate([s0, dop, aolp],axis=0) for s0,dop,aolp in zip(s0_imgs, dop_imgs, aolp_imgs)]
                imageio.mimwrite(os.path.join(save_dir, '{}_pol_cues_{}.mp4'.format(outbase, post_fix)), pol_cues_imgs, fps=args.fps, quality=8)
    
            if args.material_edit in ['swap_rg','swap_bg','only_b']:
                imageio.mimwrite(os.path.join(save_dir, '{}_mat_edit_{}.mp4'.format(outbase, post_fix)), mat_edit_imgs,fps=args.fps, quality=8) 

    if args.save_images:
        if not args.disable_rgb:
            outbase = args.expname if args.outbase is None else args.outbase
            post_fix = '{}x{}_{}_{}'.format(H, W, args.num_views, args.camera_path)
            save_dir = os.path.join('out',f"{outbase}_imgs")
            os.makedirs(save_dir, exist_ok=True)
            [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_rgb.png'), rgb) 
                for idx,rgb in enumerate(rgb_imgs)]
            [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_depth.png'), depth) 
                for idx,depth in enumerate(depth_imgs)]
            [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_normal.png'), normal) 
                for idx,normal in enumerate(normal_imgs)]
            [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_alb.png'), alb) 
                for idx,alb in enumerate(alb_imgs)]
            if not args.model.only_diffuse:
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_spec.png'), spec) 
                    for idx,spec in enumerate(spec_imgs)]
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_spec_fac0.png'), spec_fac0) 
                    for idx,spec_fac0 in enumerate(spec_fac0_imgs)]
                if calc_rough:
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_rough.png'), rough) 
                        for idx,rough in enumerate(rough_imgs)]
            if args.model.polarized:
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_mix_s0.png'), s0) 
                    for idx,s0 in enumerate(s0_imgs)]
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_mix_dop.png'), dop) 
                    for idx,dop in enumerate(dop_imgs)]
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_mix_aolp.png'), aolp) 
                    for idx,aolp in enumerate(aolp_imgs)]
                if args.color_aolp:
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_color_aolp.png'), color_aolp) 
                        for idx,color_aolp in enumerate(color_aolp_imgs)]
                if not args.model.only_diffuse:
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_diff_s0.png'), s0) 
                        for idx,s0 in enumerate(diff_s0_imgs)]
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_diff_dop.png'), dop) 
                        for idx,dop in enumerate(diff_dop_imgs)]
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_diff_aolp.png'), aolp) 
                        for idx,aolp in enumerate(diff_aolp_imgs)]
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_spec_s0.png'), s0) 
                        for idx,s0 in enumerate(spec_s0_imgs)]
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_spec_dop.png'), dop) 
                        for idx,dop in enumerate(spec_dop_imgs)]
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_cues_spec_aolp.png'), aolp) 
                        for idx,aolp in enumerate(spec_aolp_imgs)]
                if args.material_edit in ['swap_rg','swap_bg','only_b']:
                    [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_mat_edit.png'), rgb) 
                        for idx,rgb in enumerate(mat_edit_imgs)]
                [imageio.imwrite(os.path.join(save_dir, f'pred_{idx:03d}_mix_rgb.png'), rgb) 
                    for idx,rgb in enumerate(mix_rgb_imgs)]

    if args.get_metrics != None:
        metrics_L = args.get_metrics.split(',')
        metrics_savepath = os.path.join(save_dir, f'metrics.txt')
        metrics_str = ""

        avg_psnr_rgb = 0.0
        avg_psnr_spec = 0.0
        avg_psnr_mixed = 0.0
        avg_ssim_rgb = 0.0
        avg_ssim_spec = 0.0
        avg_ssim_mixed = 0.0
        avg_mae_norm = 0.0

        for i in range(len(rgb_imgs)):
            if 'mask' in args.model.use_env_mlp:
                mask_i = mask_imgs[i][...,None]
                train_mask_i = dataset.object_masks[view_ids[i]].reshape(H,W,1).cpu().detach().numpy()
                mask_i = mask_i*train_mask_i
            elif args.camera_path == 'train_views':
                mask_i = dataset.object_masks[view_ids[i]].reshape(H,W,1).cpu().detach().numpy()
            else:
                raise Exception('Cannot obtain masks')
            # For white background
            # mask_zeros_fn = lambda x: (x*mask_i + (1-mask_i)*0.).astype('uint8')
            mask_zeros_fn = lambda x: (x*mask_i).astype('uint8')

            view_id = view_ids[i]
            target_rgb = integerify(to_space_np(dataset.rgb_images[view_id].reshape(H,W,3).cpu().numpy())) 

            if args.data.type == 'Ours':
                pred_rgb = mix_rgb_imgs[i]
            elif args.data.type == 'Mitsuba2':
                pred_rgb = rgb_imgs[i]
            else:
                raise Exception('Invalid data type while computing metrics')
            if 'psnr' in metrics_L:
                # print(target_rgb.shape, pred_rgb.shape)
                from scipy.io import savemat
                savemat("out/%s_imgs/rgb_ims.mat" % outbase, {"target": mask_zeros_fn(target_rgb), "pred": mask_zeros_fn(pred_rgb)})
                psnr_rgb = calculate_psnr(mask_zeros_fn(target_rgb)/255., mask_zeros_fn(pred_rgb)/255., mask_i)
                metrics_str += "PSNR for rgb image %d: %g\n" % (view_id, psnr_rgb)
                avg_psnr_rgb += psnr_rgb

            if 'mse' in metrics_L:
                mse_rgb = mse(mask_zeros_fn(target_rgb), mask_zeros_fn(pred_rgb))
                metrics_str += "MSE for rgb image %d: %g\n" % (view_id, mse_rgb)

            if 'ssim' in metrics_L:
                ssim_rgb = ssim(mask_zeros_fn(target_rgb), mask_zeros_fn(pred_rgb), multichannel=True)
                metrics_str += "SSIM for rgb image %d: %g\n" % (view_id, ssim_rgb)
                avg_ssim_rgb += ssim_rgb

            per_pix_mse = lambda x,y: np.mean((x - y)**2, axis=2)
            per_pix_mse_rgb = per_pix_mse(target_rgb.astype(float), pred_rgb.astype(float)).astype(np.uint8)

            mySaveIm(os.path.join(save_dir, f'per_pix_err_{view_id:03d}_rgb'), per_pix_mse_rgb) 
            imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_rgb.png'), mask_fn(target_rgb)) 

            if hasattr(dataset, 'specular_images'):
                target_spec = integerify(to_space_np(dataset.specular_images[view_id].reshape(H,W,3).cpu().numpy()))
                target_mix = np.clip(linear_rgb_to_srgb_np(srgb_to_linear_rgb_np(target_rgb) + srgb_to_linear_rgb_np(target_spec)),0,255).astype(np.uint8)
                target_norm = dataset.normal_images[view_id].reshape(H,W,3).cpu().numpy()
                target_norm_orig = 0.5+0.5*(target_norm)
                target_norm /= np.linalg.norm(target_norm, ord=2, axis=2, keepdims=True)
                target_norm *= mask_i

                pred_spec = spec_imgs[i]
                pred_mix = np.clip(linear_rgb_to_srgb_np(srgb_to_linear_rgb_np(pred_rgb) + srgb_to_linear_rgb_np(pred_spec)),0,255).astype(np.uint8)
                pred_norm = normal_imgs_orig[i]
                pred_norm /= np.linalg.norm(pred_norm, ord=2, axis=2, keepdims=True)
                pred_norm *= mask_i

                if 'psnr' in metrics_L:
                    psnr_spec = calculate_psnr(mask_zeros_fn(target_spec)/255., mask_zeros_fn(pred_spec)/255.,mask_i)
                    psnr_mix = calculate_psnr(mask_zeros_fn(target_mix)/255., mask_zeros_fn(pred_mix)/255.,mask_i)
                    metrics_str += "PSNR for specular image %d: %g\n" % (view_id, psnr_spec)
                    metrics_str += "PSNR for mixed image %d: %g\n" % (view_id, psnr_mix)
                    avg_psnr_spec += psnr_spec
                    avg_psnr_mixed += psnr_mix 

                if 'mse' in metrics_L:
                    mse_spec = mse(mask_zeros_fn(target_spec), mask_zeros_fn(pred_spec))
                    mse_mix = mse(mask_zeros_fn(target_mix), mask_zeros_fn(pred_mix))
                    metrics_str += "MSE for specular image %d: %g\n" % (view_id, mse_spec)
                    metrics_str += "MSE for mixed image %d: %g\n" % (view_id, mse_mix)

                if 'ssim' in metrics_L:
                    ssim_spec = ssim(mask_zeros_fn(target_spec), mask_zeros_fn(pred_spec), multichannel=True)
                    ssim_mix = ssim(mask_zeros_fn(target_mix), mask_zeros_fn(pred_mix), multichannel=True)
                    metrics_str += "SSIM for specular image %d: %g\n" % (view_id, ssim_spec)
                    metrics_str += "SSIM for mixed image %d: %g\n" % (view_id, ssim_mix)
                    avg_ssim_spec += ssim_spec 
                    avg_ssim_mixed += ssim_mix

                per_pix_mse_spec = per_pix_mse(target_spec.astype(float), pred_spec.astype(float)).astype(np.uint8)
                per_pix_mse_mix = per_pix_mse(target_mix.astype(float), pred_mix.astype(float)).astype(np.uint8)
                imageio.imwrite(os.path.join(save_dir, f'pred_{view_id:03d}_mixed.png'), pred_mix) 
                imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_mixed.png'), mask_fn(target_mix))
                imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_specular.png'), mask_fn(target_spec)) 

                normal_truth_write = integerify(target_norm_orig)
                imageio.imwrite(os.path.join(save_dir, f'gt_{view_id:03d}_normal.png'), mask_fn(normal_truth_write)) 
                mySaveIm(os.path.join(save_dir, f'per_pix_err_{view_id:03d}_spec'), per_pix_mse_spec) 
                mySaveIm(os.path.join(save_dir, f'per_pix_err_{view_id:03d}_mixed'), per_pix_mse_mix) 


                per_pix_ang_err = lambda x, y: np.arccos(np.sum(np.multiply(x, y), axis=2))
                per_pix_ang_err_norm = np.rad2deg(per_pix_ang_err(target_norm, pred_norm))
                mean_ang_err = np.mean(np.nan_to_num(per_pix_ang_err_norm))
                metrics_str += "Mean angular error for image %d: %g\n" % (view_id, mean_ang_err)
                avg_mae_norm += mean_ang_err
                mySaveIm(os.path.join(save_dir, f'per_pix_ang_err_{view_id:03d}_normals'), per_pix_ang_err_norm)

        avg_psnr_rgb /= len(rgb_imgs)
        avg_psnr_spec /= len(rgb_imgs)
        avg_psnr_mixed /= len(rgb_imgs)
        avg_ssim_rgb /= len(rgb_imgs)
        avg_ssim_spec /= len(rgb_imgs)
        avg_ssim_mixed /= len(rgb_imgs)
        avg_mae_norm /= len(rgb_imgs)        

        metrics_str += "Average RGB PSNR: %g\n" % avg_psnr_rgb
        metrics_str += "Average SPEC PSNR: %g\n" % avg_psnr_spec
        metrics_str += "Average MIX PSNR: %g\n" % avg_psnr_mixed
        metrics_str += "Average RGB SSIM: %g\n" % avg_ssim_rgb
        metrics_str += "Average SPEC SSIM: %g\n" % avg_ssim_spec
        metrics_str += "Average Mix SSIM: %g\n" % avg_ssim_mixed
        metrics_str += "Average Normals MAE: %g\n" % avg_mae_norm

        with open(metrics_savepath, "w") as f:
            f.write(metrics_str)


if __name__ == "__main__":
    # Arguments
    # "./configs/neus.yaml"
    parser = io_util.create_args_parser()
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--render_mesh", type=str, default=None, help='the mesh ply file to be rendered')
    parser.add_argument("--device", type=str, default='cuda', help='render device')
    parser.add_argument("--downscale", type=float, default=4)
    parser.add_argument("--rayschunk", type=int, default=25)
    parser.add_argument("--camera_path", type=str, default="train_views", 
        help="choose between [spiral, interpolation, small_circle, great_circle, spherical_spiral, train_views, all_train_views]")
    parser.add_argument("--camera_inds", type=str, help="params for generating camera paths", 
        default='1,2')
    parser.add_argument("--alpha_vals", type=str, help="roughness parameters list", default='0.04')
    parser.add_argument("--get_metrics", type=str, default="psnr,ssim")
    parser.add_argument("--load_pt", type=str, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--H_scale", type=float, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--W_scale", type=float, default=None)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--disable_rgb", action='store_true')
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--alter_radiance", type=str, default=None, help='alter the radiance net with another trained ckpt.')
    parser.add_argument("--outbase", type=str, default=None, help='base of output filename')
    parser.add_argument("--use_surface_render", type=str, default=None, help='choose between [sphere_tracing, root_finding]. \n\t Use surface rendering instead of volume rendering \n\t NOTE: way faster, but might not be the original model behavior')
    parser.add_argument("--gen_env_map", type=bool, default=False)
    parser.add_argument("--diffuse_only", type=bool, default=True)
    parser.add_argument("--save_gt",type=bool, default=True, help='Plot the ground truth views as well')
    parser.add_argument("--mask_obj",type=bool,default=True)
    parser.add_argument("--save_images", type=bool, default=True, help='Save individual frames as images')
    parser.add_argument("--save_videos", type=bool, default=False, help='Save Videos')
    parser.add_argument("--render_pol", type=bool, default=True, help='Polarized rendering when the model is unpolarized')
    parser.add_argument("--color_aolp", type=bool, default=False, help='Polarized rendering when the model is unpolarized')
    parser.add_argument("--relight_map", type=str, default=None, help='Path to image file for relighting with new environment map')
    parser.add_argument("--material_edit", type=str, default=None, help='Path to image file for relighting with new environment map')
    parser.add_argument("--extract_mesh", type=bool,default=False)
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)