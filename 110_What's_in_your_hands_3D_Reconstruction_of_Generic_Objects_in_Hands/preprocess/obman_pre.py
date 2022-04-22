
# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
"""Usage:  python -m preprocess.obman_pre --data_dir PATH_TO_DATA [--out_dir xxx --vis_dir xxx]"""
import argparse
import imageio
import glob
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import tqdm
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex
from pytorch3d.transforms import Transform3d, Rotate, Translate
from pytorch3d.structures import Meshes
from nnutils.hand_utils import ManopthWrapper
from nnutils import image_utils, mesh_utils, geom_utils


def patch_segs():
    """add hand only mask as 3th channel"""
    split_list = args.split.split(',')
    pose_wrapper = ManopthWrapper().to(device)
    for split in split_list:
        index_list = load_index(split)
        for i, index in tqdm.tqdm(enumerate(index_list), total=len(index_list)):
            save_file = os.path.join(save_dir, split, 'segms_plus', '%s.png' % index)
            if os.path.exists(save_file) and skip:
                continue
            fname = os.path.join(save_dir, split, 'meta_plus', '%s.pkl' % index)
            with open(fname, 'rb') as fp:
                meta_info = pickle.load(fp)
            segms = np.array(Image.open(os.path.join(data_dir, split, 'segm', '%s.png' % index)))
            H, W, _ = segms.shape
            hand_only = extract_hand_only_mask(pose_wrapper, meta_info['cTh'], meta_info['hA'], H, os.path.join(save_dir, index))
            segms[..., 2] = hand_only
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            imageio.imwrite(save_file, segms)

            if i > 2:
                break


def extract_hand_only_mask(hand_wrapper, cTh, hA, H, prefix):
    """
    : param cTh: (4,4)?
    : param hA: (45, )
    : return: numpy of (H, W, ) in scale 0, 100
    """
    device = 'cuda'
    cTh = geom_utils.matrix_to_se3(torch.FloatTensor([cTh]).to(device))
    hA = torch.FloatTensor([hA]).to(device)

    cHand, _ = hand_wrapper(cTh, hA)
    cameras = PerspectiveCameras(3.75, device=device)
    iHand = mesh_utils.render_mesh(cHand, cameras, out_size=H)

    mask = iHand['mask'][0, 0].cpu().detach().numpy()
    mask = (mask * 100).clip(0, 255).astype(np.uint8)
    return mask


def patch_meta():
    split_list = args.split.split(',')
    pose_wrapper = ManopthWrapper().to(device)
    for split in split_list:
        index_list = load_index(split)
        for i, index in tqdm.tqdm(enumerate(index_list), total=len(index_list)):
            save_file = os.path.join(save_dir, split, 'meta_plus', '%s.pkl' % index)
            if os.path.exists(save_file) and skip:
                continue
            fname = os.path.join(data_dir, split, 'meta', '%s.pkl' % index)
            with open(fname, 'rb') as fp:
                meta_info = pickle.load(fp)

            global_rt, art_pose, obj_pose = extract_rt_pose(meta_info, pose_wrapper)
            meta_info['cTh'] = global_rt
            meta_info['hA'] = art_pose
            meta_info['cTo'] = obj_pose

            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'wb') as fp:
                pickle.dump(meta_info, fp)
            if i < 2:
                vis_meta(meta_info, pose_wrapper, split, index)


def glob_all(data_dir):
    index_list = glob.glob(osp.join(data_dir, 'meta/*.pkl'))
    index_list = [osp.basename(e).split('.')[0] for e in index_list]
    return index_list


def vis_meta(meta_info, pose_wrapper, split, index):
    device = 'cuda'
    prefix = os.path.join(vis_dir, '%s_%s' % (split, index))

    image = imageio.imread(os.path.join(data_dir, split, 'rgb', '%s.jpg' % index))
    canvas = torch.FloatTensor([image.transpose([2, 0, 1])]).to(device)
    canvas = F.adaptive_avg_pool2d(canvas, [224, 224]) / 255 * 2 - 1

    f = 480 / 128  # 3.75
    cameras = PerspectiveCameras(f, device=device)

    oObj = load_obj(device, None, meta_info["class_id"], meta_info["sample_id"])
    # cTo = Transform3d(device=device, matrix=torch.FloatTensor([meta_info['cTo']]).transpose(1, 2).to(device))
    cTo = torch.FloatTensor([meta_info['cTo']]).to(device)
    cTo = geom_utils.se3_to_matrix(geom_utils.matrix_to_se3(cTo))
    cTo = Transform3d(device=device, matrix=cTo.transpose(1, 2))
    cObj = mesh_utils.apply_transform(oObj, cTo)

    cTh = torch.FloatTensor([meta_info['cTh']]).to(device)
    se3 = geom_utils.matrix_to_se3(cTh)
    art_pose = torch.FloatTensor([meta_info['hA']]).to(device)
    cHand_param, _ = pose_wrapper(se3, art_pose, )
    scene = mesh_utils.join_scene([cHand_param, cObj])

    image = mesh_utils.render_mesh(scene, cameras)
    image_utils.save_images(image['image'], prefix + '_para',
                            mask=image['mask'], bg=canvas, r=0.9)
    image_utils.save_images(canvas, prefix + '_inp')


def load_index(split):
    return [line.strip() for line in open(os.path.join(data_dir, '%s.txt' % split))]


def extract_rt_pose(meta_info, pose_wrapper):
    device = 'cuda'
    cam_extr = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0, 0, 0, 1]
        ]
    ).astype(np.float32)
    cTw = Transform3d(device=device, matrix=torch.FloatTensor([cam_extr]).transpose(1, 2).to(device))
    wTo = Transform3d(device=device, matrix=torch.FloatTensor([meta_info['affine_transform']]).transpose(1, 2).to(device))
    cTo = wTo.compose(cTw)
    cTo_mat = cTo.get_matrix().transpose(-1, -2).cpu().detach().numpy()[0]

    # src mesh
    zeros = torch.zeros([1, 3], device=device, dtype=torch.float32)
    art_pose = torch.FloatTensor([meta_info['pca_pose']]).to(device)
    art_pose = pose_wrapper.pca_to_pose(art_pose) - pose_wrapper.hand_mean
    hHand, _ = pose_wrapper(None, art_pose, zeros, zeros, mode='inner')
    # dst mesh
    wVerts = torch.FloatTensor([meta_info['verts_3d']]).to(device)
    textures = TexturesVertex(torch.ones_like(wVerts)).to(device)
    wHand = Meshes(wVerts, pose_wrapper.hand_faces, textures)

    wTh = solve_rt(hHand.verts_padded(), wHand.verts_padded())  
    cTh = wTh.compose(cTw)
    cTh_mat = cTh.get_matrix().transpose(-1, -2).cpu().detach().numpy()[0]
    art_pose_npy = art_pose.cpu().detach().numpy()[0]

    return cTh_mat, art_pose_npy, cTo_mat



def solve_rt(src_mesh, dst_mesh):
    """
    (N, P, 3), (N, P, 3)
    """
    device = src_mesh.device
    src_centroid = torch.mean(src_mesh, -2, keepdim=True)
    dst_centroid = torch.mean(dst_mesh, -2, keepdim=True)
    src_bar = (src_mesh - src_centroid)
    dst_bar = (dst_mesh - dst_centroid)
    cov = torch.bmm(src_bar.transpose(-1, -2), dst_bar)
    u, s, v = torch.svd(cov)
    vh = v.transpose(-1, -2)
    rot_t = torch.matmul(u, vh)
    rot = rot_t.transpose(-1, -2)  # v, uh

    trans = dst_centroid - torch.matmul(src_centroid, rot_t)  # (N, 1, 3)?

    rot = Rotate(R=rot_t, device=device)
    trans = Translate(trans.squeeze(1), device=device)

    rt = rot.compose(trans)

    return rt


def load_obj(device, scale, cls, instance):
    fname = os.path.join(shape_dir, cls, instance, 'models', 'model_normalized.obj')
    return mesh_utils.load_mesh(fname).to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--no_skip", action='store_true')
    parser.add_argument("--segm", action='store_true')
    parser.add_argument("--split", default='test,val,train', type=str)
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--vis_dir", default=None, type=str)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    device = 'cuda:0'
    args = parse_args()

    vis_dir = '/checkpoint/yufeiy2/hoi_output/obman/'
    skip = not args.no_skip
    data_dir = args.data_dir + '/obman/'
    shape_dir = args.data_dir + '/obmanobj/'
    save_dir = args.out_dir if args.out_dir is not None else data_dir
    vis_dir = args.vis_dir if args.vis_dir is not None else save_dir
    patch_meta()
    if args.segm:
        patch_segs()