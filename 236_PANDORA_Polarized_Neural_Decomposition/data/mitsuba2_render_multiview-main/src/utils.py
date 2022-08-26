import os
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# From https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/utils.py
from functools import wraps
def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

def normalize(v):
    import torch
    return torch.nn.functional.normalize(v, dim=-1)

def normalize_np(v):
    return  v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),
                     1e-12)

def spec_srgb_lin(c, toLin=True, fac=2.2):
    import torch
    import torch.nn.functional as F
    if toLin:
        # return torch.exp(F.leaky_relu(c))-1
        # return torch.exp(F.relu(c))-1.
        # return torch.exp(c)-1.
        #Soft plus
        # return torch.log(1+torch.exp(5*c))
        # return 0.04*(torch.exp(c)-1.)
        # return 0.1*(torch.exp(c)-1.)
        # return 0.1*torch.exp(2*c)
        # return F.softplus(c)**fac
        # return c
        return torch.clamp(c, min=1e-12)**(fac)
        # return torch.where(c>0,c**fac,c)
    else:
        return torch.clamp(c, min=1e-12)**(1./fac)

def linear_rgb_to_srgb_ub(c):
    # Unbounded without clipping
    import torch
    import torch.nn.functional as F
    # Original definition of srgb space. Causes issues with differentiation
    # if_mask = c <= 0.0031308
    # return_if = 12.92*c.abs()
    
    # return_else = 1.055*(c.abs()+1e-12)**(1/2.4) - 0.055
    
    # return torch.where(if_mask, return_if, return_else)
    # return torch.where(c>0.,torch.abs(c)**(1/2.2),c)
    return torch.sign(c)*torch.abs(c)**(1/2.2)
    # return (F.sigmoid((c-0.001)*100)*c**(1/2.2))+0.01*c
def linear_rgb_to_srgb(c):
    import torch
    # Original definition of srgb space. Causes issues with differentiation
    # if_mask = c <= 0.0031308
    # return_if = 12.92*c.abs()
    
    # return_else = 1.055*(c.abs()+1e-12)**(1/2.4) - 0.055
    
    # return torch.where(if_mask, return_if, return_else)
    return torch.clamp(c,min=1e-12)**(1./2.2)

def srgb_to_linear_rgb(c):
    import torch
    # Original definition of srgb space. Causes issues with differentiation
    # if_mask = c <= 0.04045
    # return_if = (1.0/12.92)*c.abs()
    
    # return_else = ((c.abs()+0.055)*(1./1.055))**(2.4) 
    # return torch.where(if_mask, return_if, return_else)
    return torch.clamp(c, min=1e-12)**(2.2)

def linear_rgb_to_srgb_np(c):
    # Original definition of srgb space. Causes issues with differentiation
    # return np.where(c < 0.0031308, c * 12.92, 1.055*(c**(1.0 / 2.4)) - 0.055)
    return np.maximum(c,1e-12)**(1./2.2)

def srgb_to_linear_rgb_np(c):
    # Original definition of srgb space. Causes issues with differentiation
    # return np.where(c < 0.0031308, c * 12.92, 1.055*(c**(1.0 / 2.4)) - 0.055)
    return np.maximum(c,1e-12)**(2.2)

def imread_exr_ch(filename, **params):
    """
    Optional params:
    """
    #t = time.time()
    # Open the input file
    f = OpenEXR.InputFile(filename)
    #print('1: Time elapsed: %.06f'%(time.time()-t))
    
    #t =time.time()
    # Get the header (we store it in a variable because this function read the file each time it is called)
    header = f.header()
    #print('2: Time elapsed: %.06f'%(time.time()-t))
    # Compute the size
    dw = header['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1

    # Use the attribute "v" of PixelType objects because they have no __eq__
    pixformat_mapping = {Imath.PixelType(Imath.PixelType.FLOAT).v: np.float32,
                         Imath.PixelType(Imath.PixelType.HALF).v: np.float16,
                         Imath.PixelType(Imath.PixelType.UINT).v: np.uint32}

    # Get the number of channels
    nc = len(header['channels'])

    # Check the data type 
    #t = time.time()
    dtGlobal = list(header['channels'].values())[0].type
    data = {}
    for i, c in enumerate(header['channels']):
        if c in params['keys_list'][0]:
            dt = header['channels'][c].type
            data[c] = np.fromstring(f.channel(c), 
                                    dtype=pixformat_mapping[dt.v]).reshape((h, w))
    #print('3: Time elapsed: %.06f'%(time.time()-t))
    
    data_arrs = []
    
    if ('keys_list' in params):
        keys_list = params['keys_list']
    else:
        keys_list = [['R','G','B']]
    
    for keys in keys_list:
        data_arr = np.stack([data[k] for k in keys],axis=-1)
        data_arrs.append(data_arr)
    
    data_arrs = np.stack(data_arrs,axis=-1)[...,0]

    if ('output_header' in params):
        if params['output_header']:
            return data_arrs, header
        else:
            return data_arrs
    else:
        return data_arrs


def viz_cues(cues_dict, save_path):
    s0 = cues_dict['s0']
    dop = cues_dict['dop']
    aolp = cues_dict['aolp']

    os.makedirs(save_path, exist_ok=True)

    #Visualize
    plt.figure()
    plt.imshow(aolp[...,0]/180.,cmap='twilight',vmin=0.,vmax=1.)
    plt.colorbar(); plt.axis('off')
    plt.savefig(f'{save_path}cues_aolp.png',
                bbox_inches='tight',pad_inches=0);plt.close()

    plt.figure()
    plt.imshow(dop[...,0],cmap='viridis',vmin=0.,vmax=1.)
    plt.colorbar(); plt.axis('off')
    plt.savefig(f'{save_path}cues_dop.png',
                bbox_inches='tight',pad_inches=0);plt.close()

    # plt.figure()
    # plt.imshow(stokes_np[...,4],cmap='viridis')
    # plt.colorbar()
    # plt.savefig(f'{result_path}/s0.png')

    plt.figure()
    plt.imshow(linear_rgb_to_srgb_np(s0),cmap='viridis')
    plt.axis('off')
    plt.savefig(f'{save_path}cues_s0.png',
                bbox_inches='tight',pad_inches=0);plt.close()