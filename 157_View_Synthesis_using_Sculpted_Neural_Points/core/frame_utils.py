import numpy as np
import torch
from PIL import Image
import os
from os.path import *
import re
import cv2
import sys
from plyfile import PlyData, PlyElement
import imageio


import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg' or ext == '.JPG':
        return cv2.imread(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


def write_pfm(file: str, image, scale=1):
    with open(file, 'wb') as f:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)

## utf8 version
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def grayscale_visualization(im, label, vmin=None, vmax=None):
    # im should have shape H x W
    fig = plt.figure()
    plt.imshow(im, vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)
    plt.title('mean value: %.3f' % np.mean(im))
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return data

def save_ply(plyfilename, vertexs, vertex_colors):
    # vert pos has shape N x 3
    # vert_colors has shape N x 3
    # save
    if torch.is_tensor(vertexs):
        vertexs = vertexs.cpu().numpy()
    if torch.is_tensor(vertex_colors):
        vertex_colors = ((vertex_colors.cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)[...,[2,1,0]]

    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)

def load_ply(plyfilename):
    data = PlyData.read(plyfilename)
    vertices = data['vertex']
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])

    print('loading ply file with %d points' % len(x))

    xyz = np.stack([x, y, z], axis=1) # N x 3

    return xyz

def load_ply_color(plyfilename):
    data = PlyData.read(plyfilename)
    vertices = data['vertex']
    r = np.array(vertices['red'])
    g = np.array(vertices['green'])
    b = np.array(vertices['blue'])

    brg = np.stack([b, g, r], axis=1) # N x 3, bgr, range [0,255], uint8

    return brg

def make_animation(model, video_name, val_dataset, ref_intrinsics, logger, rasterize_rounds=5):
    model.eval()
    metrics = {}

    if not os.path.exists('./saved_videos'):
        os.mkdir('./saved_videos')

    # make a video with varying cam pose
    render_poses = val_dataset.get_render_poses()
    render_viewpose= val_dataset.get_render_poses(radius=20) # larger movement

    N_views = render_poses.shape[0]

    # pre-select subset to reduce flickering
    num_pts_to_keep = round(model.vert_pos.shape[1] * (1.0 - model.pts_dropout_rate))
    pts_id_to_keep = torch.multinomial(torch.ones_like(model.vert_pos[0]), num_pts_to_keep, replacement=False)

    all_frames = []
    with torch.no_grad():
        for i_batch in range(N_views):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_pose = render_poses[i_batch:i_batch+1] # 1 x 4 x 4

            start.record()
            rgb_est = model.evaluate(None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=pts_id_to_keep)  # 1 x 3 x H x W
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('total render time for one image:', start.elapsed_time(end))

            all_frames.append(rgb_est)


    all_frames = torch.cat(all_frames)
    logger.summ_rgbs('animation/motion', all_frames, fps=20, force_save=True)

    all_frames = (all_frames + 1.0) / 2.0
    all_frames = torch.clamp(all_frames, 0.0, 1.0)
    all_frames = all_frames[:, [2, 1, 0]]  # bgr2rgb, N x 3 x H x W, range [0,1]
    all_frames = all_frames.permute(0, 2, 3, 1).cpu().numpy()  # N x H x W x 3, range [0,1]

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    # imageio.mimwrite(os.path.join('./saved_videos/%s.mp4' % video_name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)
    imageio.mimwrite(os.path.join('./saved_videos/%s.gif' % video_name), to8b(all_frames), fps=20) # lower quality but no codec required
    # imageio.mimwrite(os.path.join('./saved_videos/%s.wmv' % video_name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)

    # # make a video with fixed cam pose and varying lighting dir
    # all_frames = []
    # with torch.no_grad():
    #     for i_batch in range(N_views):
    #         start = torch.cuda.Event(enable_timing=True)
    #         end = torch.cuda.Event(enable_timing=True)

    #         target_viewpose = render_viewpose[i_batch:i_batch+1] # 1 x 4 x 4
    #         # target_pose = render_poses[int(3*N_views/4):int(3*N_views/4)+1] # 1 x 4 x 4
    #         # target_pose = render_poses[0:1]  # 1 x 4 x 4
    #         target_pose = render_poses[int(N_views/4):int(N_views/4)+1]  # 1 x 4 x 4

    #         start.record()
    #         rgb_est = model.evaluate(None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=pts_id_to_keep, target_viewpose=target_viewpose)  # 1 x 3 x H x W
    #         end.record()

    #         # Waits for everything to finish running
    #         torch.cuda.synchronize()

    #         print('total render time for one image:', start.elapsed_time(end))

    #         all_frames.append(rgb_est)


    # all_frames = torch.cat(all_frames)
    # logger.summ_rgbs('animation/viewdir', all_frames, fps=20, force_save=True)

    # all_frames = (all_frames + 1.0) / 2.0
    # all_frames = torch.clamp(all_frames, 0.0, 1.0)
    # all_frames = all_frames[:, [2, 1, 0]]  # bgr2rgb, N x 3 x H x W, range [0,1]
    # all_frames = all_frames.permute(0, 2, 3, 1).cpu().numpy()  # N x H x W x 3, range [0,1]

    # to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    # imageio.mimwrite(os.path.join('./saved_videos/view_%s.mp4' % video_name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)

    model.train()

    return all_frames
