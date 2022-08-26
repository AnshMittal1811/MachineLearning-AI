import sys
sys.path.append('.')
import os
from math import radians
import json
import yaml

import configargparse
import mitsuba
mitsuba.set_variant('scalar_spectral_polarized')
import numpy as np
import matplotlib.pyplot as plt
import imageio

from src.utils import  linear_rgb_to_srgb_np, normalize_np
from src.polarization import cues_from_stokes_stack_np
from mitsuba.core import Thread, ScalarTransform4f,LogLevel,Spectrum,Transform4f
from mitsuba.core.xml import load_file, load_dict

def create_scene(cam_trns_mtx,
                 cam_trns_type='lookat',
                 object_name='sphere',
                 no_aovs=True,
                 trans_factor=1.0,
                 refl_factor=1.0,
                 env_map_name="20060807_wells6_hd.hdr",
                 env_scale=4,
                 N_samples = 64,
                 N_res = 400,
                 cam_fov = 67.4,
                 opaque = False,
                 only_specular= False,
                 include_floor= False,
                 roughness_alpha=0.0,
                 obj_name='mesh.obj',
                 scale_obj=1.0,
                 rgb_val='0.1,0.1,0.1',
                 diff_texture=None,
                 show_env=True,
                 render_stokes=False,
                 ):
    if cam_trns_type == 'c2w':
        cam_trns = ScalarTransform4f(cam_trns_mtx)
    elif cam_trns_type == 'lookat':
        cam_trns = ScalarTransform4f.look_at(
                                origin=cam_trns_mtx[0],
                                target=cam_trns_mtx[1],
                                up=cam_trns_mtx[2])

    diff_rgb = [float(x) for x in rgb_val.split(',')]
    # print(diff_rgb)

    platform_bsdf = {
        "type": "blendbsdf",
        "weight": refl_factor,
        "diff_bsdf":
        {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                # "value": [0.83, 0.68, 0.21],
                # "value": [219/255., 159/255., 95/255.],
                "value": [0.25*219/255., 
                          0.25*159/255.,
                          0.25*95/255.],
            },
        },
        "spec_bsdf":
        {
            "type":"roughdielectric",
            "specular_transmittance": 0.0,
            "specular_reflectance": 1.0,
            "alpha": 0.15,
        },
    }

    if not opaque:
        bsdf =  {
                    "type":"thindielectric",
                    "specular_transmittance": trans_factor,
                    "specular_reflectance": refl_factor,
                }
        # bsdf =  {
        #             "type":"dielectric",
        #             "specular_transmittance": trans_factor,
        #             "specular_reflectance": refl_factor,
        #         }
    else:
        if diff_texture is not None:
            diff_reflectance = {
                "type": "bitmap",
                "filename": diff_texture, 
            }
        else:
            diff_reflectance = {
                "type": "rgb",
                "value": diff_rgb,
            }
        bsdf = {
            "type": "blendbsdf",
            "weight": refl_factor,
            "diff_bsdf":
            {
                "type": "diffuse",
                "reflectance": diff_reflectance,
            },
            "spec_bsdf":
            {
                "type":"roughdielectric",
                "specular_transmittance": 0.0,
                "specular_reflectance": 1.0,
                "alpha": roughness_alpha,
            },
        }
        # bsdf = {
        #         "type": "roughplastic",
        #         "diffuse_reflectance": diff_reflectance,
        #         "specular_reflectance": refl_factor,
        #         "alpha": roughness_alpha,
        # }
    if object_name == 'sphere':
        object_mesh = {
            "type": "sphere",
            "bsdf": bsdf,
            "radius": 0.75,
        }
    elif object_name == 'plane':
        object_mesh = {
             "type": "rectangle",
             "to_world": 
                  ScalarTransform4f.look_at(
                                 # origin=[0., -0.3375, 0.],
                                 origin=[0., -0.5, 0.],
                                 target=[0,0,0],
                                 up=[0,0,1]
                 ),
            "bsdf": bsdf,
        }
    else :
        object_mesh={
            "type": "obj",
            "filename": obj_name,
            "mybsdf": bsdf,
            "to_world": Transform4f.scale((scale_obj,scale_obj,scale_obj)),
        }
    if not render_stokes:
        main_int = {
            "type": "volpath",
            "max_depth": 8,
            # "hide_emitters": not show_env
            "hide_emitters": True,
        }
    else:
        main_int = {
                "type": "stokes",
                "intep": {
                    "type": "volpath",
                    "max_depth": 8,
                    # "hide_emitters": not show_env,
                    "hide_emitters": True,
                },
            }
    if no_aovs:
        integrator = main_int
    else:
        integrator= {
            "type": "aov",
            "aovs": "norm:sh_normal",
            "myintegrator1": main_int,
        }


    if env_map_name == 'constant':
        light={
            "type": "constant",
            "radiance": 0.01,
        }
    else:
        light={
            "type": "envmap",
            "filename": env_map_name,
            "scale": env_scale,
        }
    scene_dict = {
        "type": "scene",
        "myintegrator": integrator,
        "object": object_mesh, 
        "mysensor":{
            "type": "perspective",
            "to_world": cam_trns,
            "fov": cam_fov,
            "near_clip": 0.001,
            "far_clip": 100000,
            "sampler":{
                "type": "ldsampler",
                "sample_count": N_samples,
            },
            "myfilm":{
                "type":"hdrfilm",
                "width": N_res,
                "height": N_res,
            }
        },
        "light":light,
    }
    if object_name=='aug':
        scene_dict['platform_mesh']={
            "type": "obj",
            "filename": "platform.obj",
            "mybsdf": platform_bsdf,
            "to_world": Transform4f.scale((scale_obj,scale_obj,scale_obj)),
        }
    elif object_name == 'snow_globe':
        scene_dict['base_mesh'] ={
            "type": "obj",
            "filename": "snow_globe_base.obj",
            "mybsdf": platform_bsdf,
        }
        scene_dict['glass_mesh'] ={
            "type": "sphere",
            # "filename": "snow_globe_glass.obj",
            "mybsdf": {
                "type": "thindielectric"
            },
            "radius": 0.632,
        }
    if include_floor and (not only_specular) :
        scene_dict["floor"]= {
             "type": "rectangle",
             "to_world": 
                  ScalarTransform4f.look_at(
                                 # origin=[0., -0.3375, 0.],
                                 origin=[0., -0.5, 0.],
                                 target=[0,0,0],
                                 up=[0,0,1]
                 ),
         }

    scene = load_dict(scene_dict)

    return scene

def config_parser():
    parser = configargparse.ArgumentParser()
    # Views
    views_group = parser.add_argument_group(title='views')
    views_group.add_argument('--N_views', type=int,
                        default=100)
    views_group.add_argument('--is_random_views', action='store_true')
    views_group.add_argument('--is_dither_views', action='store_true')
    views_group.add_argument('--is_spiral_views', action='store_true')
    views_group.add_argument('--is_upper_views', action='store_true')
    views_group.add_argument('--theta_max', type=float, default=360.)
    views_group.add_argument('--seed', type=int, default=100)
    views_group.add_argument('--cam_depth', type=float, default=1.5)

    # Scene
    render_group = parser.add_argument_group(title='render')
    render_group.add_argument('--object_name', type=str,
                        default='sphere')
    render_group.add_argument('--no_aovs', action='store_true')
    render_group.add_argument('--trans_factor', type=float, default=1.0)
    render_group.add_argument('--refl_factor', type=float, default=0.0)
    render_group.add_argument('--N_samples', type=int, default=64)
    render_group.add_argument('--N_res', type=int,
                        default=400)
    render_group.add_argument('--cam_fov', type=float,
                        default=67.4)
    render_group.add_argument('--opaque',action='store_true')
    render_group.add_argument('--cam_trns_type', type=str,
                        default='lookat',help='options: c2w, lookat')
    render_group.add_argument('--only_specular', action='store_true')
    render_group.add_argument('--env_map_name', type=str,
                        default='20060807_wells6_hd.hdr',help='env map filename ')
    render_group.add_argument('--env_scale', type=float,
                        default=4.,help='env map scaling ')
    render_group.add_argument('--obj_name', type=str,
                        default='mesh.obj',help='env map filename ')
    render_group.add_argument('--include_floor', action='store_true')
    render_group.add_argument('--roughness_alpha', type=float,
                        default=0.0)
    render_group.add_argument('--scale_obj', type=float,
                        default=1.0)
    render_group.add_argument('--rgb_val', type=str,
                        default='0.15,0.04,0.10',help='env map filename ')
    render_group.add_argument('--diff_texture', type=str,
                        default=None,help='filename for diffuse texture')
    render_group.add_argument('--show_env', type=bool,
                        default=False,help='Show env map as bg in render')
    render_group.add_argument('--render_stokes', type=bool,
                        default=False,help='Whether to perform polarized rendering')
                    

    # IO
    io_group = parser.add_argument_group(title='io')
    io_group.add_argument('--config', is_config_file=True,
                        help='Config file')
    io_group.add_argument('--scenes_dir', type=str,
                        default='data/mitsuba2_scenes/')
    io_group.add_argument('--render_name', type=str,
                        default='random')
    io_group.add_argument('--result_dir', type=str,
                        default='data/mitsuba2_renders/')
    io_group.add_argument('--plot_cues',action='store_true')
    io_group.add_argument('--save_format',type=str,
                          default='neurecon', help='Options: neurecon, nerf,neuralpil,physg')
    io_group.add_argument('--debug',action='store_true')
    
    return parser

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    return c2w


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def lookat_from_rot(theta, phi, r):
    # phi wrt xz plane
    # theta wrt positive z axis
    origin = np.array([r*np.cos(phi)*np.sin(theta) ,
                       r*np.sin(phi),
                       r*np.cos(phi)*np.cos(theta)])
    target = np.array([0.,0.,0.])
    up = np.array([0.,1.,0.])
    lookat_mtx = np.stack([origin,target,up],0)
    return lookat_mtx

def P_from_lookat(lookat_mtx,HWf):
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    # Format as mentioned in
    # https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md

    origin = lookat_mtx[[0]]
    target = lookat_mtx[[1]]
    up =     lookat_mtx[[2]]

    l = normalize_np(target-origin)
    s = normalize_np(np.cross(l,up))
    u1 = np.cross(s,l)
    
    R = np.concatenate([s,u1,-l],0)
    t = -R@origin.T
    E = np.concatenate([R,t],-1)

    H,W,f = HWf
    K = np.array([[f, 0., W/2.],
                  [0., f, H/2.],
                  [0., 0., 1]])
    P = K@E # 3x4
    # Convert origin of 2D coordinates from bottom left to top left
    P = np.array([[1, 0., 0.],
                  [0., -1, H],
                  [0., 0., 1.]])@P # 3x4


    # [0 0 0 1] concatenated
    P = np.concatenate([P,
                np.array([[0.,0.,0.,1]])],0) # 4x4 

    # Convert conventions
    # Mitsuba X1 Up: +y Towards camera: +z Right handed
    # DTU dataset Y1: Up: -z Towards camera: +x Right handed
    # Conversion Matrix
    C = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0.,  0.],
                  [0., 0., 0.,  1.]])
    P = P@C
    return P
def E_K_from_lookat_neuralpil(lookat_mtx,HWf):
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    # Format as mentioned in
    # https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md

    origin = lookat_mtx[[0]]
    target = lookat_mtx[[1]]
    up =     lookat_mtx[[2]]

    l = normalize_np(target-origin)
    s = normalize_np(np.cross(l,up))
    u1 = np.cross(s,l)
    
    R = np.concatenate([s,u1,-l],0)
    t = -R@origin.T
    E = np.concatenate([R,t],-1)

    H,W,f = HWf
    K = np.array([[f, 0., W/2.,0.],
                  [0., f, H/2.,0.],
                  [0., 0., 1,  0.],
                  [0., 0., 0,  1.],])
    # Convert origin of 2D coordinates from bottom left to top left
    # P = np.array([[1, 0., 0.],
    #               [0., -1, H],
    #               [0., 0., 1.]])@P # 3x4


    # [0 0 0 1] concatenated
    E = np.concatenate([E,
                np.array([[0.,0.,0.,1]])],0) # 4x4 

    # Convert conventions
    # Mitsuba X1 Up: +y Towards camera: +z Right handed
    # DTU dataset Y1: Up: -z Towards camera: +x Right handed
    # Conversion Matrix
    # C = np.array([[0., -1., 0., 0.],
    #               [0., 0., -1., 0.],
    #               [1., 0., 0.,  0.],
    #               [0., 0., 0.,  1.]])
    # C = np.array([[-1., 0., 0., 0.],
    #               [0., 0., 1., 0.],
    #               [0., 1., 0.,  0.],
    #               [0., 0., 0.,  1.]])
    # E = E@C

    return E,K

def E_K_from_lookat(lookat_mtx,HWf):
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    # Format as mentioned in
    # https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md

    origin = -lookat_mtx[[0]]
    target = lookat_mtx[[1]]
    up =     lookat_mtx[[2]]

    l = normalize_np(target-origin)
    s = normalize_np(np.cross(l,up))
    u1 = np.cross(s,l)
    
    R = np.concatenate([s,u1,-l],0)
    # t = -R@origin.T
    t = R@origin.T
    E = np.concatenate([R,t],-1)

    H,W,f = HWf
    K = np.array([[-f, 0., W/2.,0.],
                  [0., -f, H/2.,0.],
                  [0., 0., 1,  0.],
                  [0., 0., 0,  1.],])
    # Convert origin of 2D coordinates from bottom left to top left
    # P = np.array([[1, 0., 0.],
    #               [0., -1, H],
    #               [0., 0., 1.]])@P # 3x4


    # [0 0 0 1] concatenated
    E = np.concatenate([E,
                np.array([[0.,0.,0.,1]])],0) # 4x4 

    # Convert conventions
    # Mitsuba X1 Up: +y Towards camera: +z Right handed
    # DTU dataset Y1: Up: -z Towards camera: +x Right handed
    # Conversion Matrix
    # C = np.array([[0., -1., 0., 0.],
    #               [0., 0., -1., 0.],
    #               [1., 0., 0.,  0.],
    #               [0., 0., 0.,  1.]])
    # C = np.array([[-1., 0., 0., 0.],
    #               [0., 0., 1., 0.],
    #               [0., 1., 0.,  0.],
    #               [0., 0., 0.,  1.]])
    # E = E@C

    return E,K

def main():
    # Get arg groups
    parser = config_parser()
    args = parser.parse_args()
    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in    group._group_actions}
        arg_groups[group.title]=configargparse.Namespace(**group_dict) 

    # IO
    alpha_str = str(args.roughness_alpha)
    alpha_str = alpha_str.replace('.', 'p')
    result_path = f'{args.result_dir}/{args.object_name}_{args.render_name}_a={alpha_str}/'
    os.makedirs(result_path, exist_ok=True)
    if args.config is not None:
        f = os.path.join(result_path,'config.txt')
        with open(f, 'w') as file:
            # file.write(open(args.config, 'r').read()) # Only saves config.txt
            yaml.dump(args.__dict__,file,default_flow_style=False)
    # Add the scene directory to the FileResolver's search path
        Thread.thread().file_resolver().append(f'{args.scenes_dir}')
    if args.object_name not in ['sphere','plane']:
        Thread.thread().file_resolver().append(f'{args.scenes_dir}/{args.object_name}/')
    # Silence logs
    Thread.thread().logger().set_log_level(LogLevel.Error)


    # If specular only set the trans_factor and refl_factor correctly
    if args.only_specular:
        arg_groups['render'].trans_factor = 0.
        arg_groups['render'].refl_factor = 1.
        arg_groups['render'].show_env = False

    if not args.only_specular:
        img_folder = 'image'
    else:
        img_folder = 'specular'
    # Out file
    if args.save_format == 'nerf':
        out_data = {'camera_angle_x':args.cam_fov}
        out_data['frames'] = []
    else:
        cam_dict = {}
    if args.save_format in ['neurecon','neuralpil','nerf','physg']:
        os.makedirs(f'{result_path}/{img_folder}/',exist_ok=True)
        os.makedirs(f'{result_path}/normal/',exist_ok=True)
        os.makedirs(f'{result_path}/mask/',exist_ok=True)
        if args.save_format == 'neuralpil':
            os.makedirs(f'{result_path}/masks/',exist_ok=True)
            os.makedirs(f'{result_path}/images/',exist_ok=True)
    

    # Iterate over views
    stepsize = args.theta_max/args.N_views
    np.random.seed(args.seed)
    for i in range(0,args.N_views):
        # render_name = f'r_{i:03d}'
        render_name = f'{i:04d}'
        if args.is_random_views:
            theta = np.random.uniform(0, args.theta_max/360.)*360.
            if args.is_upper_views:
                phi = np.random.uniform(0,1)*90
            else:
                phi = np.random.uniform(0,1)*180 - 90
        elif args.is_dither_views:
            # alpha_range = np.linspace(0,2*np.pi,args.N_views)
            # phi_range = 5+5*np.cos(alpha_range)
            # theta_range = 340+5*np.sin(alpha_range)
            # phi = phi_range[i]
            # theta = theta_range[i]
            # phi_range = np.linspace(0.,50.,1)+1.
            phi_range = np.linspace(25.,50.,3)+1.
            # theta_range = np.linspace(-65.,65.,7)+1.
            # theta_range = np.linspace(-180.,60,15)+1.
            theta_range = np.linspace(-180.,157.5,15)+1.
            phi_grid, theta_grid = np.meshgrid(phi_range,theta_range)
            phi = phi_grid.reshape(-1)[i]
            theta = theta_grid.reshape(-1)[i]
        elif args.is_spiral_views:
            #https://en.wikipedia.org/wiki/Spiral#Spherical_spirals
            c = 6
            phi_range = np.linspace(45,15,args.N_views)
            theta_range = c*phi_range
            phi = phi_range[i]
            theta = theta_range[i]

        else:
            # print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            phi = 0.
            theta = i*stepsize
            # render_name =  'r_{0:03d}'.format(int(i * stepsize))
    
        # Use rot to rotate camera
        if args.cam_trns_type == 'c2w':
            cam_mtx = pose_spherical(theta+np.pi,phi,-args.cam_depth)
            cam_mtx_save = pose_spherical(-theta,phi,args.cam_depth)
        elif args.cam_trns_type == 'lookat':
            cam_mtx = lookat_from_rot(theta*np.pi/180., phi*np.pi/180., args.cam_depth)
            focal =  .5* args.N_res / np.tan(.5*args.cam_fov*np.pi/180.) 
            if args.save_format in ['neurecon','neuralpil']:
                cam_mtx_save = P_from_lookat(cam_mtx,
                                        [args.N_res,args.N_res,focal])
            elif args.save_format in ['nerf']:
                E_save,_ = E_K_from_lookat_neuralpil(cam_mtx,
                                        [args.N_res,args.N_res,focal])
                pose_save = np.array([[-1, 0, 0, 0], 
                                      [0, 0, 1, 0], 
                                      [0, 1, 0, 0], 
                                      [0, 0, 0, 1]])@ np.linalg.inv(E_save)
            elif args.save_format in ['physg']:
                E_save,K_save = E_K_from_lookat(cam_mtx,
                                        [args.N_res,args.N_res,focal])
        # Load the scene for an XML file
        scene = create_scene(cam_mtx,
                             **arg_groups['render'].__dict__)

        # Get the scene's sensor (if many, can pick one by specifying the index)
        sensor = scene.sensors()[0]

        # Call the scene's integrator to render the loaded scene with the desired sensor
        scene.integrator().render(scene, sensor)

        # The rendered data is stored in the film
        film = sensor.film()

        # Write out data as high dynamic range OpenEXR file
        film.set_destination_file(f'{result_path}/{render_name}')
        film.develop()

        # Get Bitmap
        stokes_np = np.array(film.bitmap(),dtype=np.float32)#R,G,B,A,S0R,S0G,S0B,...

        # get normals,mask
        if args.no_aovs:
            # Get mask from alpha map. Might be wrong when show_env or not opaque
            mask = stokes_np[...,3]
        else:
            normals = stokes_np[...,4:7]
            # Normals are zero where there is no object
            mask = (normals == 0.).sum(-1) < 3
            # Coordinate change for normals to match neurecon conventions
            # Based on the C matrix defined above
            from copy import deepcopy
            normals_changed = deepcopy(normals)
            normals_changed[...,1] = -normals[...,0]
            normals_changed[...,2] = -normals[...,1]
            normals_changed[...,0] =  normals[...,2]
            normals = normals_changed

        # Get and save polarimetric cues
        if args.render_stokes:
            if args.no_aovs:
                cues = cues_from_stokes_stack_np(stokes_np[...,4:])
            else:
                cues = cues_from_stokes_stack_np(stokes_np[...,7:])

            s0 = cues['s0']
            dop = cues['dop']
            aolp = cues['aolp']
        else:
            if not args.only_specular:
                s0 = stokes_np[...,:3] 
            else:
                # To remove the background
                s0 = mask[...,None] * stokes_np[...,:3]

        if args.render_stokes and args.plot_cues:
            #Visualize
            plt.figure()
            plt.imshow(aolp[...,0],cmap='twilight',vmin=0.,vmax=1.)
            plt.colorbar(); plt.axis('off')
            plt.savefig(f'{result_path}/cues_{img_folder}_aolp_{render_name}.png',
                        bbox_inches='tight',pad_inches=0);plt.close()

            plt.figure()
            plt.imshow(dop[...,0],cmap='viridis',vmin=0.,vmax=1.)
            plt.colorbar(); plt.axis('off')
            plt.savefig(f'{result_path}/cues_{img_folder}_dop_{render_name}.png',
                        bbox_inches='tight',pad_inches=0);plt.close()

            plt.figure()
            plt.imshow(linear_rgb_to_srgb_np(s0),cmap='viridis')
            plt.axis('off')
            plt.savefig(f'{result_path}/cues_{img_folder}_s0_{render_name}.png',
                        bbox_inches='tight',pad_inches=0);plt.close()
            imageio.imwrite(f'{result_path}/normals_{render_name}.png',
                            0.5+0.5*normals)

        if args.save_format == 'nerf':
            frame_data = {
                # Changed for neuralpil blender
                # 'file_path': f'../neurecon/{result_path}/image/{render_name}.exr',
                'file_path': f'{render_name}',
                'rotation': radians(stepsize),
                'transform_matrix': listify_matrix(pose_save)
            }
            out_data['frames'].append(frame_data)
        elif args.save_format in ['neurecon','neuralpil']:
            cam_dict[f'world_mat_{i}'] = cam_mtx_save
            cam_dict[f'scale_mat_{i}'] = np.eye(4)
        elif args.save_format in ['physg']:
            cam_dict[f"{i:04d}.exr"] = {
                "K": list(K_save.reshape(-1)),
                "W2C": list(E_save.reshape(-1)),
                "img_size": [args.N_res,args.N_res]
            }
        if args.save_format in ['neurecon', 'neuralpil','nerf','physg']:
            # Saving in neurecon format
            imageio.imwrite(f'{result_path}/{img_folder}/{i:04d}.exr',
                            s0)
            if not args.only_specular:
                imageio.imwrite(f'{result_path}/normal/{i:04d}.exr',
                                normals)
                imageio.imwrite(f'{result_path}/mask/{i:04d}.png',
                            mask[...,None]*np.array([1,1,1])[None,None])
            if args.save_format == 'neuralpil':
                imageio.imwrite(f'{result_path}/images/{i:04d}.jpg',
                                (np.clip(linear_rgb_to_srgb_np(s0),0,1.)*255.).astype('uint8'))
                imageio.imwrite(f'{result_path}/masks/{i:04d}.jpg',
                            mask[...,None]*np.array([1,1,1])[None,None])
        if args.debug:
            import pdb; pdb.set_trace()


    if args.save_format == 'nerf':
        # Save json
        with open(result_path + '/' + 'transforms_train.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
        with open(result_path + '/' + 'transforms_test.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
        with open(result_path + '/' + 'transforms_val.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
    elif args.save_format == 'neurecon':
        np.savez(f'{result_path}/cameras.npz',**cam_dict)
    elif args.save_format == 'physg':
        with open(result_path + '/' + 'cam_dict_norm.json', 'w') as out_file:
            json.dump(cam_dict, out_file, indent=4)
    
    if args.render_stokes and args.plot_cues:
        # Create videos
        import subprocess
        for file_name in [f'cues_{img_folder}_{x}' for x in ['dop','s0','aolp']]: 
            subprocess.call(f'''ffmpeg -y -i {result_path}/{file_name}_r_%03d.png'''
                            f''' -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" ''' 
                            f''' -c:v libx264 -pix_fmt yuv421p '''
                            f''' -framerate 0.5'''
                            f''' {result_path}/videos_{file_name}.mp4''',
                            shell=True)



if __name__ == "__main__":
    main()