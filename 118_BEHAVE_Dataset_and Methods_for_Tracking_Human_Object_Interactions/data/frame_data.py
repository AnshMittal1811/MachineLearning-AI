"""
loads more data for each frame
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
sys.path.append(os.getcwd())
from os.path import join, isfile
from psbody.mesh import Mesh
import json
import cv2
import numpy as np
import pickle as pkl
from data.sync_frame import KinectFrameReader
from data.seq_utils import SeqInfo


class FrameDataReader(KinectFrameReader):
    "read more data: pc, mocap, fitted smpl, obj etc"
    def __init__(self, seq, empty=None, ext='jpg', check_image=True):
        seq_info = SeqInfo(seq)
        super(FrameDataReader, self).__init__(seq, empty, seq_info.kinect_count(), ext, check_image=check_image)
        self.seq_info = seq_info
        self.kids = self.seq_info.kids

    def get_pc(self, idx, cat='person', convert=False):
        pcfile = self.get_pcpath(idx, cat, convert)
        if not isfile(pcfile):
            pcfile = self.get_pcpath(idx, cat, not convert)
        return self.load_mesh(pcfile)

    def get_pcpath(self, idx, cat, convert=False):
        if cat=='person':
            name = 'person'
        else:
            name = self.seq_info.get_obj_name(convert)
        frame_folder = self.get_frame_folder(idx)
        pcfile = join(frame_folder, f'{name}/{name}.ply')
        return pcfile

    def load_mesh(self, pcfile):
        if not isfile(pcfile):
            return None
        m = Mesh()
        m.load_from_file(pcfile)
        return m

    def get_J3d(self, idx):
        frame_folder = self.get_frame_folder(idx)
        pcfile = join(frame_folder, "person/person_J.ply")
        return self.load_mesh(pcfile)

    def get_mocap_mesh(self, idx, kid=1):
        mocap_file = self.get_mocap_meshfile(idx, kid)
        return self.load_mesh(mocap_file)

    def get_mocap_meshfile(self, idx, kid=1):
        frame_folder = self.get_frame_folder(idx)
        mocap_file = join(frame_folder, f'k{kid}.mocap.ply')
        return mocap_file

    def get_mocap_pose(self, idx, kid=1):
        jsonfile = join(self.get_frame_folder(idx), 'k{}.mocap.json'.format(kid))
        if not isfile(jsonfile):
            return None
        params = json.load(open(jsonfile))
        return np.array(params['pose'])

    def get_mocap_beta(self, idx, kid=1):
        jsonfile = join(self.get_frame_folder(idx), 'k{}.mocap.json'.format(kid))
        if not isfile(jsonfile):
            return None
        params = json.load(open(jsonfile))
        return np.array(params['betas'])

    def get_smplfit(self, idx, save_name, ext='ply'):
        if save_name is None:
            return None
        mesh_file = self.smplfit_meshfile(idx, save_name, ext)
        return self.load_mesh(mesh_file)

    def smplfit_meshfile(self, idx, save_name, ext='ply'):
        mesh_file = join(self.get_frame_folder(idx), 'person', save_name,
                         f'person_fit.{ext}')
        return mesh_file

    def objfit_meshfile(self, idx, save_name, ext='ply', convert=True):
        name = self.seq_info.get_obj_name(convert=convert)
        mesh_file = join(self.get_frame_folder(idx), name, save_name,
                         f'{name}_fit.{ext}')
        if not isfile(mesh_file):
            name = self.seq_info.get_obj_name()
            mesh_file = join(self.get_frame_folder(idx), name, save_name,
                         f'{name}_fit.{ext}')
        return mesh_file

    def get_objfit(self, idx, save_name, ext='ply'):
        if save_name is None:
            return None
        mesh_file = self.objfit_meshfile(idx, save_name, ext)
        return self.load_mesh(mesh_file)

    def objfit_param_file(self, idx, save_name):
        name = self.seq_info.get_obj_name(convert=True)
        pkl_file = join(self.get_frame_folder(idx), name, save_name,
                         f'{name}_fit.pkl')
        return pkl_file

    def get_objfit_params(self, idx, save_name):
        "return angle and translation"
        if save_name is None:
            return None, None
        pkl_file = self.objfit_param_file(idx, save_name)
        if not isfile(pkl_file):
            return None, None
        fit = pkl.load(open(pkl_file, 'rb'))
        return fit['angle'], fit['trans']

    def get_smplfit_params(self, idx, save_name):
        "return pose, beta, translation"
        if save_name is None:
            return None, None, None
        pkl_file = self.smplfit_param_file(idx, save_name)
        if not isfile(pkl_file):
            return None, None, None
        fit = pkl.load(open(pkl_file, 'rb'))
        return fit['pose'], fit['betas'], fit['trans']

    def smplfit_param_file(self, idx, save_name):
        return join(self.get_frame_folder(idx), 'person', save_name, 'person_fit.pkl')

    def times2indices(self, frame_times):
        "convert frame time str to indices"
        indices = [self.get_frame_idx(f) for f in frame_times]
        return indices

    def get_body_j3d(self, idx):
        file = self.body_j3d_file(idx)
        if not isfile(file):
            return None
        data = json.load(open(file))
        J3d = np.array(data["body_joints3d"]).reshape((-1, 4))  # the forth column is the score
        return J3d

    def body_j3d_file(self, idx):
        pcfile = self.get_pcpath(idx, 'person')
        return pcfile.replace(".ply", "_J3d.json")

    def get_body_kpts(self, idx, kid, tol=0.5):
        J2d_file = join(self.get_frame_folder(idx), f'k{kid}.color.json')
        if not isfile(J2d_file):
            return None
        data = json.load(open(J2d_file))
        J2d = np.array(data["body_joints"]).reshape((-1, 3))
        J2d[:, 2][J2d[:, 2] < tol] = 0
        return J2d

    def get_mask(self, idx, kid, cat='person', ret_bool=True):
        if cat=='person':
            file = join(self.get_frame_folder(idx), f'k{kid}.person_mask.{self.ext}')
            # print(file)
        elif cat == 'obj':
            file = join(self.get_frame_folder(idx), f'k{kid}.obj_rend_mask.jpg')
            if not isfile(file):
                file = join(self.get_frame_folder(idx), f'k{kid}.obj_mask.{self.ext}')
        else:
            raise NotImplemented
        if not isfile(file):
            return None
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if ret_bool:
            mask = mask > 127
        return mask

    def get_person_mask(self, idx, kids, ret_bool=True):
        frame_folder = join(self.seq_path, self.frames[idx])
        mask_files = [join(frame_folder, f'k{k}.person_mask.{self.ext}') for k in kids]
        # print(mask_files)
        masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]
        if ret_bool:
            masks = [x > 127 if x is not None else None for x in masks]
        return masks

    def get_pcfiles(self, frames, cat, convert=False):
        # if cat !='person':
        #     name = self.seq_info.get_obj_name(convert)
        # else:
        #     name = 'person'
        pcfiles = [self.get_pcpath(x, cat, convert) for x in frames]
        return pcfiles

    def pc_exists(self, idx, cat, convert=False):
        pcfile = self.get_pcpath(idx, cat, convert)
        return isfile(pcfile)

    def cvt_end(self, end):
        batch_end = len(self) if end is None else end
        if batch_end>len(self):
            batch_end = len(self)
        return batch_end



