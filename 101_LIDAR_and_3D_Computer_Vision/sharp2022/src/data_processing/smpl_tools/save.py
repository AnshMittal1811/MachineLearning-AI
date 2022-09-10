from display_utils import display_model
from label import get_label
import sys
import os
import re
from tqdm import tqdm
import numpy as np
import pickle

sys.path.append(os.getcwd())


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_pic(res, smpl_layer, file, logger, dataset_name, target):
    _, _, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/picture/{}".format(dataset_name, file_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    for i in tqdm(range(Jtr.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=True)
    logger.info('Pictures saved')


def save_params(res, file, out_dir):
    path = os.path.normpath(file)
    gt_file_name = path.split(os.sep)[-2]
    full_file_name = path.split(os.sep)[-1][:-4]
    pose_params, shape_params, verts, Jtr = res
    create_dir_not_exist(os.path.join(out_dir, gt_file_name))
    print('Saving params at {}'.format(out_dir))
    pose_params = (pose_params.cpu().detach()).numpy().tolist()
    shape_params = (shape_params.cpu().detach()).numpy().tolist()
    Jtr = (Jtr.cpu().detach()).numpy().tolist()
    verts = (verts.cpu().detach()).numpy().tolist()
    params = {}
    params["pose_params"] = pose_params
    params["shape_params"] = shape_params
    params["Jtr"] = Jtr
    with open(os.path.join((out_dir),
                           "{}/{}_smpl_params.pkl".format(gt_file_name, full_file_name)), 'wb') as f:
        pickle.dump(params, f)
