import numpy as np
import os, imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json
import cv2
import argparse

from plyfile import PlyData, PlyElement

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def write_pfm(file, image, scale=1):
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

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            







#####################

def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def get_w2c(images):
    poses = []
    for i in images:
        # print(dir(images[i]))
        print(images[i].name)
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        # c2w = np.linalg.inv(w2c)
        poses.append(w2c)
    return np.array(poses) # N x 4 x 4

def colmap_dense2dtu(basedir, target_size=(1600, 1200), rescale=False):
    w, h = target_size

    ply_file = Path(basedir) / 'scene.ply'
    target_dir = Path(basedir) / 'DTU_format'
    img_target_dir = target_dir / 'images'
    depth_target_dir = target_dir / 'depths'
    depth4_target_dir = target_dir / 'depths_4'
    vis_target_dir = target_dir / 'depth_vis'
    cam_file = target_dir / 'cameras.npz'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(depth_target_dir):
        os.makedirs(depth_target_dir)
    if not os.path.exists(img_target_dir):
        os.makedirs(img_target_dir)
    if not os.path.exists(vis_target_dir):
        os.makedirs(vis_target_dir)
    if not os.path.exists(depth4_target_dir):
        os.makedirs(depth4_target_dir)

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    img_path = Path(basedir) / 'images'
    raw_imgs = [os.path.join(img_path, f) for f in sorted(os.listdir(img_path)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    raw_shape = imageio.imread(raw_imgs[0]).shape
    size_origin = (raw_shape[1], raw_shape[0])

    img_scale = size_origin[0] / target_size[0]
    assert abs(size_origin[1] / target_size[1] - img_scale) < 1e-2

    cameras = read_cameras_binary(Path(basedir) / 'sparse' / '0' / 'cameras.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    # get intrinsics
    focal, cx, cy = cameras[1].params

    K = np.eye(3)
    K[0, 0] = K[1,1] = focal
    K[0, 2] = cx
    K[1, 2] = cy

    K[:2, :] /= img_scale


    images = {k: v for k, v in sorted(images.items(), key=lambda item: item[1].name)}

    poses = get_w2c(images) # w2c



    depths, masks = load_colmap_dense(basedir)
    # N x H x W and N x H x W, respectively

    if rescale:
        # re-scale so that the near bound disparity is around 0.0025
        min_depth = np.min(depths[masks])
        max_depth = np.max(depths[masks])

        Dnear_original = 1.0 / min_depth
        Dnear_target = 0.9 * .0025

        depth_scale = Dnear_target / Dnear_original

        depths[masks] = np.clip(depths[masks], min_depth, max_depth)

        depths = depths / depth_scale
        poses[:, :3, 3] = poses[:, :3, 3] / depth_scale

        min_depth = np.min(depths[masks])
        max_depth = np.max(depths[masks])

        print('min/max disparity after scaling: %.4f/%.4f' % (1. / max_depth, 1. / min_depth))

    cam_info = {}
    cam_info['intrinsics'] = K
    cam_info['extrinsics'] = poses
    np.savez(cam_file, **cam_info)

    for id_im in range(1, len(images) + 1):
        # load and resize images
        img = cv2.imread(raw_imgs[id_im - 1])
        img = cv2.resize(img, target_size)
        cv2.imwrite(str(img_target_dir / (str(id_im).zfill(8) + '.jpg')), img)

        depth = depths[id_im - 1]
        mask = masks[id_im - 1]

        depth[np.logical_not(mask)] = -1.0

        write_pfm(depth_target_dir / (str(id_im).zfill(8) + '.pfm'), depth)

        depth_vis = np.zeros_like(depth)

        depth_vis[depth < 0] = 0.0
        zfar = np.max(depth)
        znear = 0.0

        depth_vis[depth >= 0] = 0.9 * depth[depth >= 0] / (zfar - znear) + 0.1
        depth_vis = (depth_vis * 255.0).astype(np.uint8)

        cv2.imwrite(str(vis_target_dir / (str(id_im).zfill(8) + '.jpg')), depth_vis)

    return



def save_ply(plyfilename, vert_pos, vert_colors):
    # vert pos has shape N x 3
    # vert_colors has shape N x 3
    # save
    vertexs = vert_pos
    vertex_colors = vert_colors.astype(np.uint8)
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



def load_colmap_dense(datadir, H=None, W=None):
    depths = []
    masks = []

    image_list = sorted(os.listdir(os.path.join(datadir, "images")))

    ply_path = os.path.join(datadir, 'dense', 'fused.ply')
    ply_masks = read_ply_mask(ply_path)

    for image_name in image_list:
        depth_path = os.path.join(datadir, 'dense/stereo/depth_maps', image_name + '.geometric.bin')
        depth = read_array(depth_path)
        mask = ply_masks[image_name]
        if H is not None:
            depth = cv2.resize(depth, (W, H))
            mask = cv2.resize(mask, (W, H))
        depths.append(depth)
        masks.append(mask > 0.5)

    return np.stack(depths), np.stack(masks)

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def load_point_vis(path, masks):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        print('point number: {}'.format(n))
        for i in range(n):
            m = struct.unpack('<I', f.read(4))[0]
            for j in range(m):
                idx, u, v = struct.unpack('<III', f.read(4 * 3))
                masks[idx][v, u] = 1

def read_ply_mask(path):
    images_bin_path = os.path.join(os.path.dirname(path), 'sparse', 'images.bin')
    images = read_images_binary(images_bin_path)
    names = [dd[1].name for dd in images.items()]
    shapes = {}
    for name in names:
        depth_fname = os.path.join(os.path.dirname(path), 'stereo', 'depth_maps', name + '.geometric.bin')
        shapes[name] = read_array(depth_fname).shape

    ply_vis_path = path + '.vis'
    assert os.path.exists(ply_vis_path)
    masks = [np.zeros(shapes[name], dtype=np.uint8) for name in names]
    load_point_vis(ply_vis_path, masks)
    return {name: mask for name, mask in zip(names, masks)}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="scene name", type=str, required=True)
    parser.add_argument("--colmap_output_dir", type=str, required=True)
    args = parser.parse_args()
    
    scene_name = args.scene
    print('processing scene %s' % scene_name)
    colmap_dense2dtu(os.path.join(args.colmap_output_dir, scene_name))