import json
import PIL
import imageio
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
  
from joblib.parallel import Parallel, delayed
import argparse
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.transforms.transform3d import Translate
from torchvision.transforms.transforms import Compose, Resize
from tqdm import tqdm
from PIL import Image
from pytorch3d.renderer.cameras import PerspectiveCameras
from torchvision.transforms import ToTensor, ToPILImage
import os
import os.path as osp
import pandas as pd
import pickle
import torch
import numpy as np
# from datasets.ho3d import crop_cam, proj3d, get_K, minmax, square_bbox

from nnutils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o,  cvt_axisang_t_o2i
from nnutils import mesh_utils, image_utils, geom_utils


data_dir = '/checkpoint/yufeiy2/datasets/HO3D/'
shape_dir = '~/hoi/data/ho3dobj'
vis_dir = '/checkpoint/yufeiy2/hoi_output/vis_HO3D/'
save_dir = '/checkpoint/yufeiy2/hoi_output/vis_ho3d_eval/'


def get_inp(index):
    image_dir = osp.join(data_dir, '{}', '{}', 'rgb', '{}.jpg')
    image = PIL.Image.open(image_dir.format(*index))
    trans = Compose([Resize(224), ToTensor()])
    inp = trans(image)
    return inp


def load_index_list(split, th=5):
    df = pd.read_csv(osp.join(data_dir, split + '.csv'))
    sub_df = df[df['dist'] < 5]
    print(len(df), '-->', len(sub_df))
    index_list = sub_df['index']
    folder_list = sub_df['split']
    tup_list = []
    for index, folder in zip(index_list, folder_list):
        index = (folder, index.split('/')[0], index.split('/')[1])
        tup_list.append(index)
    return tup_list


def load_anno(index, hand_wrapper, device='cpu', meta_folder='meta_plus', root=None):
    """

    Args:
        index ([type]): [description]
        hand_wrapper ([type]): [description]

    Returns:
        cHand, cObj, anno
    """
    hand_wrapper = hand_wrapper.to(device)
    # hand_gt, obj_gt ,_ = load_anno(index, hand_wrapper)
    meta_dir = os.path.join(data_dir, '{}', '{}', meta_folder, '{}.pkl')
    shape_dir = os.path.join('../hoi/data', 'ho3dobj/models', '{}', 'textured_simple.obj')

    meta_path = meta_dir.format(*index)
    with open(meta_path, "rb") as meta_f:
        anno = pickle.load(meta_f)

    pose = torch.FloatTensor(anno['handPose'])[None]  # handTrans
    trans = torch.FloatTensor(anno['handTrans'].reshape(3))[None]
    if root == 'zero':
        trans = torch.zeros_like(trans)
    hA = pose[..., 3:]
    rot = pose[..., :3]
    rot, trans = cvt_axisang_t_i2o(rot, trans)
    wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)

    wTo = geom_utils.axis_angle_t_to_matrix(
        torch.FloatTensor([anno['objRot'].reshape(3)]), 
        torch.FloatTensor([anno['objTrans'].reshape(3)]))
    hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

    rot = torch.FloatTensor([[[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]])
    cTw = geom_utils.rt_to_homo(rot, )
    cTh = cTw @ wTh
    cTo = cTh @ hTo
    anno['cTh'] = cTh[0].cpu().detach().numpy()
    anno['cTo'] = cTo[0].cpu().detach().numpy()

    cHand, cJoints = hand_wrapper(
        geom_utils.matrix_to_se3(cTh).to(device), 
        hA.to(device))
    fname = shape_dir.format(anno['objName'])
    oObj = mesh_utils.load_mesh(fname).to(device)
    cObj = mesh_utils.apply_transform(oObj, cTo.to(device))

    cam_intr = torch.FloatTensor(anno['camMat'])
    anno['cam_f'], anno['cam_p'] = image_utils.screen_intr_to_ndc_fp(cam_intr, 1, 1)

    if root == 'root':
        trans = Translate(-cJoints[:, 0], device=device)
        cObj = mesh_utils.apply_transform(cObj, trans)
        cHand = mesh_utils.apply_transform(cHand, trans)

    return cHand, cObj, anno


def get_vid2obj(vid2frame):
    vid2obj = {}
    obj2vid = {}

    for vid in vid2frame:
        for frame, folder in vid2frame[vid]:
            if vid in vid2obj:
                break
        meta_dir = os.path.join(data_dir, '{}', '{}', 'meta', '{}.pkl')            
        with open(meta_dir.format(folder, vid, frame), 'rb') as fp:
            anno = pickle.load(fp)
        obj = anno['objName']
        
        vid2obj[vid] = obj
        if obj not in obj2vid:
            obj2vid[obj] = []
        obj2vid[obj].append(vid)
    return vid2obj, obj2vid



def get_vid2frame(split_list=['train', 'val', 'test']):
    vid2frame = {}
    for split in split_list:
        folder = 'evaluation' if split == 'evaluation' else 'train'
        index_list = [line.strip() for line in open(osp.join(data_dir, '%s.txt' % split))]
        for index in index_list:
            vid, frame = index.split('/')
            if vid not in vid2frame:
                vid2frame[vid] = []
            vid2frame[vid].append((frame, folder))
    return vid2frame


def create_split():
    "index,vid,frame,dist,split(original folder),obj"
    vid2frame_folder = get_vid2frame()
    vid2obj, obj2vid = get_vid2obj(vid2frame_folder)
    
    device = 'cuda:0'
    hand_wrapper = ManopthWrapper().to(device)
    for split in VID_SPLIT:
        data_list = []
        for vid in tqdm(VID_SPLIT[split]):
            frame_list = vid2frame_folder[vid]
            for frame, folder in tqdm(frame_list):
                index = osp.join(vid, frame)
                dist = closest_dist(index, folder, hand_wrapper, device)
                data = {
                    'index': index, 
                    'vid': vid,
                    'frame': frame,
                    'dist': dist,
                    'split': folder,
                    'obj': vid2obj[vid],
                }
                data_list.append(data)
        df = pd.DataFrame(data_list)
        df.to_csv(osp.join(save_dir, '%s_vid.csv' % split))

        sub_df = sub_df.iloc[::10, :]
        sub_df.to_csv(osp.join(save_dir, '%s_vid3fps.csv' % split))


def parse_args():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('--skip', action='store_true')
    return arg_parser


def homo_meta():
    # gt: meta_gt
    # pred: meta_plus
    # evaluation set: link meta_plus = meta_pred
    split = 'evaluation'
    for vid in tqdm(os.listdir(osp.join(data_dir, split))):
        src = osp.join(data_dir, split, vid, 'meta_plus')
        dst = osp.join(data_dir, split, vid, 'meta_gt')
        link(src, dst)
    
    split = 'train'
    for vid in tqdm(os.listdir(osp.join(data_dir, split))):
        src = osp.join(data_dir, split, vid, 'meta')
        dst = osp.join(data_dir, split, vid, 'meta_gt')
        link(src, dst)


def link(src, dst):
    if osp.exists(dst):
        cmd = 'rm %s' % dst
        os.system(cmd)
    cmd = 'ln -s %s %s' % (src, dst)
    print(cmd)
    os.system(cmd)
    


def patch_meta(split):
    if split == 'evaluation':
        folder = 'evaluation'
        # meta_plus = 'meta_plus'
        meta_plus = 'meta_frank'
        det_template = osp.join('../data/homan/ho3d', 'preprocess/samples/{}/det.pkl')
    elif split == 'train':
        folder = 'train'
        det_template = osp.join('../data/homan/ho3d', 'train/samples/{}/det.pkl')
        # meta_plus = 'meta_plus'
        meta_plus = 'meta_frank'
    elif split == 'val':
        folder = 'train'
        det_template = osp.join('../data/homan/ho3d', 'val/samples/{}/det.pkl')
        meta_plus = 'meta_frank'

    device = 'cuda:0'
    hand_wrapper = ManopthWrapper().to(device)


    index_file = osp.join(data_dir, '%s.txt' % folder)
    index_list = [line.strip() for line in open(index_file)]
    
    vid_index = {}
    for index in index_list:
        vid, frame = index.split('/')
        if vid not in vid_index:
            vid_index[vid] = []
        vid_index[vid].append(frame)

    np.random.seed(123)
    vid_list = sorted(list(vid_index.keys()))

    for v, vid in tqdm(enumerate(vid_list), total=len(vid_list)):        
        for i, frame in tqdm(enumerate(vid_index[vid]), total=len(vid_index[vid])):
            index = osp.join(vid, frame)
            meta_dir = os.path.join(data_dir, folder, '{}', 'meta', '{}.pkl')
            meta_plus_dir = os.path.join(save_dir, folder, '{}', meta_plus, '{}.pkl')

            vid_idx, frame_idx = index.split('/')
            fname = osp.join(meta_plus_dir.format(vid_idx, frame_idx))
            # if osp.exists(fname):
                # continue
            
            with open(osp.join(meta_dir.format(vid_idx, frame_idx)), 'rb') as fp:
                anno = pickle.load(fp)
    
            anno = change_anno(anno, det_file)

            fname = osp.join(meta_plus_dir.format(vid_idx, frame_idx))
            os.makedirs(osp.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fp:
                pickle.dump(anno, fp)


def patch_anno(anno, pose, trans, meta, hand_wrapper):
    device = 'cpu'
    rot = torch.FloatTensor(pose)[..., :3][None]
    trans = torch.FloatTensor(trans)[None]
    glb_trans = torch.FloatTensor(meta['translations']).squeeze(1)
    glb_rot = torch.FloatTensor(meta['rotations'])

    rot, trans = cvt_axisang_t_i2o(rot, trans)
    wpThp = geom_utils.axis_angle_t_to_matrix(rot, trans)
    cTwp = geom_utils.rt_to_homo(glb_rot, glb_trans)
    
    wTc = torch.FloatTensor([[[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
            ]]).to(device)
    wThp = wTc @ cTwp @ wpThp
    rot, trans, _ = geom_utils.matrix_to_axis_angle_t(wThp)
    rot, trans = cvt_axisang_t_o2i(rot, trans)
    
    if anno['handPose'] is not None:
        # rot = anno['handPose'][..., :3]
        # pose = np.concatenate([rot, pose[..., 3:]], -1)

        pose = np.concatenate([rot.cpu().detach().numpy()[0], pose[..., 3:]], -1)
        anno['handPose'] = pose

    else:
        pose = np.concatenate([rot.cpu().detach().numpy()[0], pose[..., 3:]], -1)

        anno['handPose'] = pose
        anno['handTrans'] = trans.cpu().detach().numpy()[0]

    if len(anno['handJoints3D']) != 21:
        anno['handJoints3D'] = forward_joints(anno, hand_wrapper)
    
    return anno


def forward_joints(anno, hand_wrapper):
    device = 'cuda:0'
    pose = torch.FloatTensor([anno['handPose']]).to(device)
    trans = torch.FloatTensor([anno['handTrans']]).to(device)

    _, j3d_head = hand_wrapper(None, pose[..., 3:], pose[..., :3], trans, mode='inner')
    reorder = [
        0, 13, 14, 15, 16,
        1, 2, 3, 17,
        4, 5, 6, 18,
        10, 11, 12, 19, 
        7, 8, 9, 20,  
    ]
    inv_dict = {}
    for i, j in enumerate(reorder):
        inv_dict[j] = i
    inverse_reorder = []
    for j in range(21):
        inverse_reorder.append(inv_dict[j])
    j3d_head = j3d_head.cpu().detach().numpy()[0]
    j3d_head = j3d_head[inverse_reorder]
    return j3d_head


def load_ho_pred(anno, i, hand_wrapper):
    person_parameters = anno['person_parameters'][i]
    hand_mean = hand_wrapper.hand_mean.cpu().detach().numpy().reshape(-1)
    axisang = torch.FloatTensor(person_parameters['mano_rot'])
    rot = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)
    cali = torch.diag_embed(torch.FloatTensor([1, -1, -1]), dim1=-2, dim2=-1)
    axisang_after = geom_utils.matrix_to_axis_angle(cali @ rot)
    pose = np.concatenate([
        # axisang_after.cpu().detach().numpy(),
        axisang.cpu().detach().numpy(),
        person_parameters['mano_pose'] + hand_mean
        ], -1)[0]

    trans =  person_parameters['mano_trans'][0]

    cam_verts = person_parameters['camverts'][0]
    meta = person_parameters
    return pose, trans, meta



def filter_contact(split, skip=False):
    device = 'cpu'
    hand_wrapper = ManopthWrapper().to(device)
    # split = 'val'  # 'evaluation
    folder = 'evaluation' if split == 'evaluation' else 'train'
    
    index_file = osp.join(data_dir, '%s.txt' % split)
    index_list = [line.strip() for line in open(index_file)]


    annotations_list = Parallel(n_jobs=16 ,verbose=5)(
        delayed(closest_dist)(
            index, folder, hand_wrapper, device, skip
        )
        for i, index in tqdm(enumerate(index_list), total=len(index_list))
    )

    # print(osp.join(data_dir, '%s.csv' % split))
        

def filter_join(split):
    annotations_list = []
    folder = 'evaluation' if split == 'evaluation' else 'train'
    index_list = [line.strip() for line in open(osp.join(data_dir, '%s.txt' % split))]

    def load_one(index):
        tmp_file = osp.join(data_dir, 'tmp', folder, index + '.pkl')
        with open(tmp_file, 'rb') as fp:
            anno = pickle.load(fp)
        return anno
    annotations_list = Parallel(n_jobs=16 ,verbose=5)(
        delayed(load_one)(index)
        for i, index in tqdm(enumerate(index_list), total=len(index_list))
    )

    data = pd.DataFrame(
            annotations_list).loc[:, ["index", "vid", "frame", "dist"]]
    data.to_csv(osp.join(data_dir, '%s.csv' % split))
    print(osp.join(data_dir, '%s.csv' % split))
        

def closest_dist(index, folder, hand_wrapper, device='cpu', skip=False):
    vid_idx, frame_idx = index.split('/')

    shape_temp = os.path.join(shape_dir, 'models', '{}', 'textured_simple.obj')
    if folder == 'evaluation':
        meta_dir = os.path.join(data_dir, folder, '{}', 'meta_plus', '{}.pkl')
    else:
        meta_dir = os.path.join(data_dir, folder, '{}', 'meta', '{}.pkl')

    vid_idx, frame_idx = index.split('/')
    with open(osp.join(meta_dir.format(vid_idx, frame_idx)), 'rb') as fp:
        anno = pickle.load(fp)

    fname = shape_temp.format(anno['objName'])
    mesh = mesh_utils.load_mesh(fname, scale_verts=1).to(device)
    
    pose = torch.FloatTensor(anno['handPose'][None]).to(device)  # handTrans
    trans = torch.FloatTensor(anno['handTrans'][None]).to(device)

    hA = pose[..., 3:]
    rot = pose[..., :3]
    rot, trans = cvt_axisang_t_i2o(rot, trans)
    wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)

    wTo = geom_utils.axis_angle_t_to_matrix(
        torch.FloatTensor([anno['objRot'].reshape(3)]), 
        torch.FloatTensor([anno['objTrans'].reshape(3)])).to(device)
    hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

    rot = torch.FloatTensor([[[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]]).to(device)
    cTw = geom_utils.rt_to_homo(rot, )
    cTh = cTw @ wTh

    cHand, _ = hand_wrapper(geom_utils.matrix_to_se3(cTw @ wTh), hA)        
    
    cTo = cTh @ hTo
    cMesh = mesh_utils.apply_transform(mesh, cTo)
    cObj = sample_points_from_meshes(cMesh)  # (1 M 3)
    cHand = sample_points_from_meshes(cHand) # (1 K 3)
    dist = torch.min(((cObj - cHand.transpose(0, 1)) ** 2).sum(-1).sqrt())  # K M 3
    
    dist = dist.item() * 1000

    return dist



VID_SPLIT = {
    'train': [
        'SMu41', 'MDF12', 'BB12', 'SM2', 
        'ABF14', 'MDF14', 'SiBF12', 'SB11', 
        'BB14', 'SM4', 'SM5', 'AP13', 'GPMF13', 
        'MPM12', 'SiBF13', 'SMu40', 'ABF13', 
        'MC4', 'SiBF10', 'BB11', 'SB14', 
        'MC2', 'ShSu13', 'SiBF11', 'MPM11', 
        'ABF12', 'AP12', 'MPM13', 'SB13',
         'MC6', 'GPMF14', 'GSF11', 'GSF13', 
         'GPMF11', 'BB13', 'GPMF12', 'SS2', 
         'MPM10', 'ShSu14', 'SS3', 'ABF11', 
         'AP11', 'SiS1', 'ND2', 'SM3', 'MDF11', 
         'MDF13', 'SiBF14', 'MPM14', 'GSF12', 
         'SMu42', 'GSF14', 'SB12', 'SB10', 
         'ShSu10', 'MC5', 'AP14', 'ShSu12',
        ], 
    'test': [
        'GSF10', 'BB10', 'AP10', 'SS1', 
        'ABF10', 'SM1', 'GPMF10', 'MDF10', 
        'MC1', 'SMu1'
        ]
    }


def save_bbox_for_mocap():
    for split in ['evaluation', 'val', 'train']:
        folder = 'evaluation' if split == 'evaluation' else 'train'
        index_list = [index.split() for index in osp.join(data_dir, '%s.txt' % split)]
        for index in index_list:
            anno = 
            
            cCorner = mesh_utils.apply_transform(torch.FloatTensor(anno['objCorners3D'][None]).cuda(), cTw)
            bboxj2d = minmax(torch.cat([proj3d(cJoints, cam_intr), proj3d(cCorner, cam_intr)], dim=1))


if __name__ == '__main__':
    # split vid
    print('split train/test videos by object')
    # create_split()
    # frankmocap
    save_bbox_for_mocap()

    

    # patch_meta('evaluation')
    # patch_meta('val')
    # patch_meta('train')
    # homo_meta()

    # filter contact
    # filter_contact('train')

    # patch meta w prediction 

    # optional get foreground mask 
    
    # patch_meta('evaluation')
    # patch_meta('val')
    # patch_meta('train')
    # filter_contact('evaluation', True)
    # filter_join('evaluation')

    # parser = parse_args()
    # parser = add_slurm_args(parser)
    # args = parser.parse_args()

    # vid_list()

    # slurm_wrapper(args, save_dir, filter_contact, {'split': 'evaluation', 'skip': args.slurm})
    # slurm_wrapper(args, save_dir, render_mask, {'args': args})
