import sys
sys.path.append('.')
import numpy as np
import imageio
import matplotlib.pyplot as plt

from src.utils import linear_rgb_to_srgb, linear_rgb_to_srgb_np, normalize, normalize_np

def cues_from_stokes(stokes):
    import torch
    if torch.is_tensor(stokes):
        sqrt, atan2 = torch.sqrt, torch.atan2
    else:
        sqrt, atan2 = np.sqrt, np.arctan2
    # Assumes last dimension is 4
    dop = sqrt((stokes[...,1:]**2).sum(-1))/stokes[...,0]
    dop[stokes[...,0]<1e-6] = 0.
    aolp = 0.5*atan2(stokes[...,2],stokes[...,1])
    aolp = (aolp%np.pi)/np.pi*180
    s0 = stokes[...,0]
    return {'dop':dop,
            'aolp':aolp,
            's0':s0}

def cues_from_stokes_stack_np(stokes_np):
    # DOP sqrt(s_1^2+s_2^2+s_3^2)/s_0
    dop = np.sqrt(stokes_np[...,[3,4,5]]**2+
                  stokes_np[...,[6,7,8]]**2
                  )/stokes_np[...,[0,1,2]]

    #AOLP 0.5*arctan(s2/s1)
    aolp = 0.5*np.arctan2(stokes_np[...,[6,7,8]],
                          stokes_np[...,[3,4,5]])
    aolp[np.isnan(aolp)] = 0
    aolp = ((aolp)%(np.pi))/(np.pi)

    cues = {}
    cues['s0'] = stokes_np[...,[0,1,2]]
    cues['dop'] = dop
    cues['aolp'] = aolp

    return cues

def colorize_cues(pol_cues, select_ch=0, gamma_s0=True, color_aolp=False):
    import torch
    device = pol_cues['s0'].device
    # Add heat map to polarization cues 
    # Select one channel for dop and aolp
    color_pol_cues = {}
    # aolp
    aolp_sel = pol_cues['aolp'][...,0]/180. # Select the red channel normalize to 1
    aolp_cmap = plt.get_cmap('twilight')
    # aolp_cmap = plt.get_cmap('hsv')
    aolp_color = aolp_cmap(aolp_sel.numpy())[...,:3]
    color_pol_cues['aolp'] = torch.Tensor(aolp_color, device=device)
    # dop
    dop_sel = pol_cues['dop'][...,select_ch] # Select the red channel
    dop_cmap = plt.get_cmap('viridis')
    dop_color = dop_cmap(dop_sel.numpy())[...,:3]
    color_pol_cues['dop'] = torch.Tensor(dop_color, device=device)
    # s0
    if gamma_s0:
        color_pol_cues['s0'] = torch.clamp(linear_rgb_to_srgb(pol_cues['s0']),
                                            min=0., max=1.)
    else:
        color_pol_cues['s0'] = torch.clamp(pol_cues['s0'],
                                           min=0., max=1.)
    if color_aolp:
        import polanalyser as pa
        import cv2
        color_pol_cues['color_aolp'] = torch.Tensor(
                                            cv2.cvtColor(
                                                pa.applyColorToAoLP(aolp_sel.numpy()*np.pi,
                                                                    value=dop_sel.numpy()), 
                                                cv2.COLOR_BGR2RGB)/255.,
                                            device=device)
    return color_pol_cues


def get_fresnel(rays_d, normal):

    import torch
    # Define helper functions
    # flag to check if torch or np
    pt = torch.is_tensor(rays_d)
    cos = torch.cos if pt else np.cos
    sin = torch.sin if pt else np.sin
    acos = torch.acos if pt else np.arccos
    sqrt = torch.sqrt if pt else np.sqrt
    normize = normalize if pt else normalize_np
    if pt:
        acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7,
                                                   max=1.-1e-7))
        dot = lambda x, y: (x*y).sum(-1, keepdim=True)
        clamp = lambda x,y: torch.clamp(x,min=y)
    else:
        acos = np.arccos
        dot = lambda x, y: (x*y).sum(-1, keepdims=True)
        clamp = lambda x,y: np.maximum(x,y)

    n = normize(normal) 
    o = normize(-rays_d)
    h = n #if train_mode else normize(i+o)

    eta = 1.5

    # reflectance
    cos_theta_d = dot(h,o)
    theta_d = acos(cos_theta_d)
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1-(sin(theta_r_1)/eta)**2,
                                1e-7))) 

    # Reflection components
    R__perp = (eta_r_1*cos(theta_r_1)-eta_r_2*cos(theta_r_2))**2\
    /clamp((eta_r_1*cos(theta_r_1)+eta_r_2*cos(theta_r_2))**2,
                 1e-7)
    R__para = (eta_r_1*cos(theta_r_2)-eta_r_2*cos(theta_r_1))**2\
    /clamp((eta_r_1*cos(theta_r_2)+eta_r_2*cos(theta_r_1))**2,
                 1e-7)

    fresnel_refl = (R__para + R__perp) / 2.
    fresnel_refl[cos_theta_d < 0] = 0

    return fresnel_refl


def stokes_from_normal_rad(rays_o, rays_d, normal, diff_rads,
                           spec_rads=None,
                           train_mode=False,
                           ret_separate = False):
    import torch
    # Args:
    #   rays, normal :(H,W,3)
    #   diff_rads, spec_rads: (H,W,3)

    # Add singleton dimension for Num_lights
    rays_o = rays_o[...,None,:]
    rays_d = rays_d[...,None,:]
    normal = normal[...,None,:]
    if diff_rads is not None:
        diff_rads = diff_rads[...,None,:]
    if spec_rads is not None: 
        spec_rads = spec_rads[...,None,:]

    # Unpack rays
    

    # Define helper functions
    # flag to check if torch or np
    pt = torch.is_tensor(rays_o)
    cos = torch.cos if pt else np.cos
    sin = torch.sin if pt else np.sin
    acos = torch.acos if pt else np.arccos
    atan2 = torch.atan2 if pt else np.arctan2
    sqrt = torch.sqrt if pt else np.sqrt
    normize = normalize if pt else normalize_np
    stack = torch.stack if pt else np.stack
    lin2srgb = linear_rgb_to_srgb if pt else linear_rgb_to_srgb_np
    if pt:
        acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7,
                                                   max=1.-1e-7))
        dot = lambda x, y: (x*y).sum(-1, keepdim=True)
        clamp = lambda x,y: torch.clamp(x,min=y)
        mask_fn = lambda x,mask: torch.where(mask, x, torch.zeros_like(x))
        cross = lambda x,y: torch.cross(x,y.broadcast_to(x.shape),dim=-1) 
        clip_max = lambda x,y:torch.clamp(x, max=y)
    else:
        acos = np.arccos
        dot = lambda x, y: (x*y).sum(-1, keepdims=True)
        clamp = lambda x,y: np.maximum(x,y)
        mask_fn = lambda x,mask: mask*x
        cross = lambda x,y: np.cross(x,y) 
        clip_max = lambda x, y: torch.minimum(x,y)

    # Helper variables    
    eta = 1.5
    n = normize(normal) 
    o = normize(-rays_d)
    h = n #if train_mode else normize(i+o)
    # n_o = normize(cross(n,o))
    # h_o = normize(cross(h,o))
    n_o = normize(n - dot(n,o)*o)
    h_o = normize(h - dot(h,o)*o)

    # Using Dirs_up in global coordinates
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    x_o = normize(stack([-o[...,1],o[...,0],0*o[...,2]],-1))
    # Cross product with local up vector of [0,0,-1]
    y_o = cross(x_o,o)
    phi_o = atan2(-dot(n_o,x_o), dot(n_o,y_o))
    psi_o = atan2(-dot(h_o, x_o), dot(h_o, y_o))
    # Using Dirs_up in local coordinates
    # phi_o = atan2(n_o[...,[0]],n_o[...,[1]])
    # psi_o = atan2(h_o[...,[0]],h_o[...,[1]]) 
    # Variables for Fresnel
    # incidence
    eta_i_1, eta_i_2 = 1.0, eta
    theta_i_1  = acos(dot(n,o))
    theta_i_2 = acos(sqrt(clamp(1-(sin(theta_i_1)/eta)**2,
                                1e-7))) 

    # exitance
    eta_o_1, eta_o_2 = eta, 1.0
    theta_o_2  = acos(dot(n,o))
    theta_o_1 = acos(sqrt(clamp(1-(sin(theta_o_2)/eta)**2,
                                1e-7))) 
    # reflectance
    theta_d = acos(dot(h,o))
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1-(sin(theta_r_1)/eta)**2,
                                1e-7))) 
    
    # Transmission components
    T_i__perp = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_1)+eta_i_2*cos(theta_i_2))**2,
                1e-7)
    T_i__perp = T_i__perp*(cos(theta_i_1)>1e-7)
    T_i__para = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_2)+eta_i_2*cos(theta_i_1))**2,
                1e-7)
    T_i__para = T_i__para*(cos(theta_i_1)>1e-7)
    T_i__plus, T_i__min = 0.5*(T_i__perp+T_i__para), 0.5*(T_i__perp-T_i__para)
    # exitance
    T_o__perp = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_1)+eta_o_2*cos(theta_o_2))**2,
                1e-7)
    T_o__para = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_2)+eta_o_2*cos(theta_o_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)

    # Reflection components
    R__perp = (eta_r_1*cos(theta_r_1)-eta_r_2*cos(theta_r_2))**2\
    /clamp((eta_r_1*cos(theta_r_1)+eta_r_2*cos(theta_r_2))**2,
                1e-7)
    R__para = (eta_r_1*cos(theta_r_2)-eta_r_2*cos(theta_r_1))**2\
    /clamp((eta_r_1*cos(theta_r_2)+eta_r_2*cos(theta_r_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)
    R__plus, R__min = 0.5*(R__perp+R__para), 0.5*(R__perp-R__para)


    # Exitant  stokes  Unpolarized illumination
    stokes_diff_fac = stack([ 1+0.*T_o__min,
                        T_o__min/T_o__plus*cos(2*phi_o),
                        -T_o__min/T_o__plus*sin(2*phi_o)],
                        -1) 
                        # (H, W, Num_lights, 1, 3)
    # if train_mode:
    #     stokes_diff_fac = stack([ 1+0.*T_o__min,
    #                           T_o__min/T_o__plus*cos(2*phi_o),
    #                          -T_o__min/T_o__plus*sin(2*phi_o)],
    #                          -1) 
    #                         # (H, W, Num_lights, 1, 3)
    # else:
    #     stokes_diff_fac = stack([ T_o__plus,
    #                               T_o__min*cos(2*phi_o),
    #                              -T_o__min*sin(2*phi_o)],
    #                              -1) 
    #                             # (H, W, Num_lights, 1, 3)
    # if not train_mode:
    #     stokes_diff_fac = stokes_diff_fac*T_i__plus[...,None]
    
    if train_mode:
        stokes_spec_fac = stack([  1+0.*R__plus,
                                R__min/R__plus*cos(2*psi_o),
                                -R__min/R__plus*sin(2*psi_o)],-1)
                            # (H,W,Num_lights,1,3)
    else:
        stokes_spec_fac = stack([  R__plus,
                                R__min*cos(2*psi_o),
                                -R__min*sin(2*psi_o)],-1)
                            # (H,W,Num_lights,1,3)
        stokes_spec_fac = clip_max(clamp(stokes_spec_fac,-1.5),1.5)
    
    # Mask the regions where angles at interface are larger than 90
    if not train_mode:
        diff_mask = dot(n,o) > 0
        diff_mask = diff_mask*(dot(n,o)>0)
        stokes_diff_fac = mask_fn(stokes_diff_fac, diff_mask[...,None])
    
    if not train_mode:
        spec_mask = dot(h,o) > 0
        spec_mask = spec_mask*(dot(h,o)>0)
        stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])

    stokes_out = 0.
    # Multiply with radiance
    if diff_rads is not None:
        stokes_out_diff = stokes_diff_fac*diff_rads[..., None] 
        stokes_out = stokes_out+ stokes_out_diff.sum(-3)
                 # (H, W, Num_lights, 3(RGB), 3(stokes))

    if spec_rads is not None:
        stokes_out_spec = stokes_spec_fac*spec_rads[..., None]
        # Mean for light sources
        stokes_out = stokes_out + stokes_out_spec.sum(-3)# (H,W,Num_lights, 3)

    debug=0
    if debug:
        cues_out = cues_from_stokes(stokes_out)
        imageio.imwrite('debug/s0.exr',cues_out['s0'].astype('float32'))

        plt.imshow(cues_out['aolp'][...,0], 
                   vmin=0, vmax=180,
                   cmap='twilight')
        plt.colorbar()
        plt.savefig('debug/aolp.png')
        plt.close()

        plt.imshow(cues_out['dop'][...,0])
        plt.colorbar()
        plt.savefig('debug/dop.png')
        plt.close()
        import pdb; pdb.set_trace()

    if (spec_rads is not None) and (diff_rads is not None)\
        and ret_separate:
            return stokes_out_diff.sum(-3), stokes_out_spec.sum(-3)
    else:
        return stokes_out


def stokes_fac_from_normal(rays_o, rays_d, normal, 
                           train_mode=False,
                           ret_spec = False,
                           clip_spec=False):
    import torch
    # Args:
    #   rays, normal :(H,W,3)
    #   diff_rads, spec_rads: (H,W,3)

    # Add singleton dimension for Num_lights
    rays_o = rays_o[...,None,:]
    rays_d = rays_d[...,None,:]
    normal = normal[...,None,:]

    # Define helper functions
    # flag to check if torch or np
    pt = torch.is_tensor(rays_o)
    cos = torch.cos if pt else np.cos
    sin = torch.sin if pt else np.sin
    acos = torch.acos if pt else np.arccos
    atan2 = torch.atan2 if pt else np.arctan2
    sqrt = torch.sqrt if pt else np.sqrt
    normize = normalize if pt else normalize_np
    stack = torch.stack if pt else np.stack
    lin2srgb = linear_rgb_to_srgb if pt else linear_rgb_to_srgb_np
    acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7,
                                               max=1.-1e-7))
    dot = lambda x, y: (x*y).sum(-1, keepdim=True)
    clamp = lambda x,y: torch.clamp(x,min=y)
    mask_fn = lambda x,mask: torch.where(mask, x, torch.zeros_like(x))
    cross = lambda x,y: torch.cross(x,y.broadcast_to(x.shape),dim=-1) 
    clip_max = lambda x,y:torch.clamp(x, max=y)

    # Helper variables    
    eta = 1.5
    n = normize(normal) 
    o = normize(-rays_d)
    h = n #if train_mode else normize(i+o)
    # n_o = normize(cross(n,o))
    # h_o = normize(cross(h,o))
    n_o = normize(n - dot(n,o)*o)
    h_o = normize(h - dot(h,o)*o)

    # Using Dirs_up in global coordinates
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    x_o = normize(stack([-o[...,1],o[...,0],0*o[...,2]],-1))
    # Cross product with local up vector of [0,0,-1]
    y_o = cross(x_o,o)
    phi_o = atan2(-dot(n_o,x_o), dot(n_o,y_o))
    psi_o = atan2(-dot(h_o, x_o), dot(h_o, y_o))
    # Using Dirs_up in local coordinates
    # phi_o = atan2(n_o[...,[0]],n_o[...,[1]])
    # psi_o = atan2(h_o[...,[0]],h_o[...,[1]]) 
    # Variables for Fresnel
    # incidence
    eta_i_1, eta_i_2 = 1.0, eta
    theta_i_1  = acos(dot(n,o))
    theta_i_2 = acos(sqrt(clamp(1-(sin(theta_i_1)/eta)**2,
                                1e-7))) 

    # exitance
    eta_o_1, eta_o_2 = eta, 1.0
    theta_o_2  = acos(dot(n,o))
    theta_o_1 = acos(sqrt(clamp(1-(sin(theta_o_2)/eta)**2,
                                1e-7))) 
    # reflectance
    theta_d = acos(dot(h,o))
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1-(sin(theta_r_1)/eta)**2,
                                1e-7))) 
    
    # Transmission components
    T_i__perp = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_1)+eta_i_2*cos(theta_i_2))**2,
                1e-7)
    T_i__perp = T_i__perp*(cos(theta_i_1)>1e-7)
    T_i__para = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_2)+eta_i_2*cos(theta_i_1))**2,
                1e-7)
    T_i__para = T_i__para*(cos(theta_i_1)>1e-7)
    T_i__plus, T_i__min = 0.5*(T_i__perp+T_i__para), 0.5*(T_i__perp-T_i__para)
    # exitance
    T_o__perp = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_1)+eta_o_2*cos(theta_o_2))**2,
                1e-7)
    T_o__para = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_2)+eta_o_2*cos(theta_o_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)

    # Reflection components
    R__perp = (eta_r_1*cos(theta_r_1)-eta_r_2*cos(theta_r_2))**2\
    /clamp((eta_r_1*cos(theta_r_1)+eta_r_2*cos(theta_r_2))**2,
                1e-7)
    R__para = (eta_r_1*cos(theta_r_2)-eta_r_2*cos(theta_r_1))**2\
    /clamp((eta_r_1*cos(theta_r_2)+eta_r_2*cos(theta_r_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)
    R__plus, R__min = 0.5*(R__perp+R__para), 0.5*(R__perp-R__para)

    # R__plus = clip_max(clamp(R__plus, 0.04),1.)
    # R__min = clip_max(clamp(R__min, 0.),0.16)

    # Exitant  stokes  Unpolarized illumination
    stokes_diff_fac = stack([ 1+0.*T_o__min,
                        T_o__min/T_o__plus*cos(2*phi_o),
                        -T_o__min/T_o__plus*sin(2*phi_o)],
                        -1) 
                        # (H, W, Num_lights, 1, 3)
    # if train_mode:
    #     stokes_diff_fac = stack([ 1+0.*T_o__min,
    #                           T_o__min/T_o__plus*cos(2*phi_o),
    #                          -T_o__min/T_o__plus*sin(2*phi_o)],
    #                          -1) 
    #                         # (H, W, Num_lights, 1, 3)
    # else:
    #     stokes_diff_fac = stack([ T_o__plus,
    #                               T_o__min*cos(2*phi_o),
    #                              -T_o__min*sin(2*phi_o)],
    #                              -1) 
    #                             # (H, W, Num_lights, 1, 3)
    # if not train_mode:
    #     stokes_diff_fac = stokes_diff_fac*T_i__plus[...,None]

    # Mask the regions where angles at interface are larger than 90
    # diff_mask = dot(n,o) > 1e-7
    # diff_mask = diff_mask*(dot(n,o)>1e-7)
    # stokes_diff_fac = mask_fn(stokes_diff_fac, diff_mask[...,None])
    
    if not ret_spec:
        return stokes_diff_fac
    else:
        stokes_spec_fac = stack([  1+0.*R__plus,
                                    R__min/R__plus*cos(2*psi_o),
                                    -R__min/R__plus*sin(2*psi_o)],-1)
                                # (H,W,Num_lights,1,3)
        
        if clip_spec:
            spec_mask = dot(h,o) > 1e-7
            spec_mask = spec_mask*(dot(h,o)>1e-7)
            stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])
            R__plus = mask_fn(R__plus, spec_mask)
        
        return stokes_diff_fac, stokes_spec_fac, R__plus[..., None]


def get_R(theta_d, eta=1.5):
    import torch
    pt = 1
    cos = torch.cos if pt else np.cos
    sin = torch.sin if pt else np.sin
    acos = torch.acos if pt else np.arccos
    atan2 = torch.atan2 if pt else np.arctan2
    sqrt = torch.sqrt if pt else np.sqrt
    normize = normalize if pt else normalize_np
    stack = torch.stack if pt else np.stack
    lin2srgb = linear_rgb_to_srgb if pt else linear_rgb_to_srgb_np
    acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7,
                                               max=1.-1e-7))
    dot = lambda x, y: (x*y).sum(-1, keepdim=True)
    clamp = lambda x,y: torch.clamp(x,min=y)
    mask_fn = lambda x,mask: torch.where(mask, x, torch.zeros_like(x))
    cross = lambda x,y: torch.cross(x,y.broadcast_to(x.shape),dim=-1) 
    clip_max = lambda x,y:torch.clamp(x, max=y)

    # Helper variables    
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1-(sin(theta_r_1)/eta)**2,
                                1e-7))) 
    R__perp = (eta_r_1*cos(theta_r_1)-eta_r_2*cos(theta_r_2))**2\
    /clamp((eta_r_1*cos(theta_r_1)+eta_r_2*cos(theta_r_2))**2,
                1e-7)
    R__para = (eta_r_1*cos(theta_r_2)-eta_r_2*cos(theta_r_1))**2\
    /clamp((eta_r_1*cos(theta_r_2)+eta_r_2*cos(theta_r_1))**2,
                1e-7)
    return R__perp, R__para

def plot_R(eta=1.5):
    import matplotlib.pyplot as plt
    import torch
    import math
    import os
    save_dir = 'viz/pol_terms_v1/'
    os.makedirs(save_dir,exist_ok=True)
    theta_d_range = torch.linspace(0,120,1000)
    cos_theta_d_range = torch.cos(theta_d_range*math.pi/180.)
    R__perp, R__para = get_R(theta_d_range*math.pi/180.,eta=eta)
    # R__perp, R__para = torch.clip(R__perp,0,1), torch.clip(R__para, 0, 1)
    R__plus, R__min = 0.5*(R__perp+R__para), 0.5*(R__perp-R__para)
    dop = torch.clip(R__min/R__plus,0,1)
    # Schlick approximation https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8_9.pdf
    R_0 = 0.04
    R_schlick = 0.04+(1-0.04)*(1 - cos_theta_d_range)**5
    R__plus_elu = R__plus*torch.nn.functional.elu(cos_theta_d_range)
    for label,val in {'R_perp':R__perp, 'R_para':R__para,
                      'R_plus':R__plus, 'R_min':R__min,
                      'dop': dop, 'R_schlick': R_schlick,
                      'R__plus_elu': R__plus_elu}.items():
        plt.figure()
        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.plot(cos_theta_d_range, val)
        plt.xlabel('cos_theta_d')
        plt.savefig(f'{save_dir}/{label}_vs_cos_theta_eta_{eta:2f}.png')
        plt.figure()
        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.plot(theta_d_range, val)
        plt.xlabel('theta_d (in degrees)')
        plt.savefig(f'{save_dir}/{label}_vs_theta_eta_{eta:2f}.png')
        plt.close()

if __name__ == '__main__':
    plot_R()