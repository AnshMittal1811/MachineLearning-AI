# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import pickle
import os.path as osp
from typing import Tuple
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, Rotate, Translate

from manopth.manolayer import ManoLayer
from manopth.tensutils import th_with_zeros, th_posemap_axisang

from nnutils import geom_utils


class ManopthWrapper(nn.Module):
    # TODO: mano
    def __init__(self, mano_path='externals/mano/', **kwargs):
        super().__init__()
        self.mano_layer_right = ManoLayer(
            mano_root=mano_path, side='right', use_pca=kwargs.get('use_pca', False), ncomps=kwargs.get('ncomps', 45),
            flat_hand_mean=kwargs.get('flat_hand_mean', True))
        self.metric = kwargs.get('metric', 1)
        
        self.register_buffer('hand_faces', self.mano_layer_right.th_faces.unsqueeze(0))
        self.register_buffer('hand_mean', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_mean']).unsqueeze(0))
        self.register_buffer('t_mano', torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, ))
        self.register_buffer('th_selected_comps', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_components']))
        self.register_buffer('inv_scale', 1. / torch.sum(self.th_selected_comps ** 2, dim=-1))  # (D, ))

        if osp.exists(osp.join(mano_path, 'contact_zones.pkl')):
            with open(osp.join(mano_path, 'contact_zones.pkl'), 'rb') as fp:
                contact = pickle.load(fp)['contact_zones']
                contact_list = []
                for ind, verts_idx in contact.items():
                    contact_list.extend(verts_idx)
                    self.register_buffer('contact_index_%d' % ind, torch.LongTensor(verts_idx))
            self.register_buffer('contact_index' , torch.LongTensor(contact_list))


    def forward(self, glb_se3, art_pose, axisang=None, trans=None, return_mesh=True, mode='outer', **kwargs) -> Tuple[Meshes, torch.Tensor]:
        N = len(art_pose)
        device = art_pose.device

        if mode == 'outer':
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            if art_pose.size(-1) == 45:
                art_pose = torch.cat([axisang, art_pose], -1)
            verts, joints, faces = self._forward_layer(art_pose, trans)

            mat_rt = geom_utils.se3_to_matrix(glb_se3)
            trans = Transform3d(matrix=mat_rt.transpose(1, 2))
            verts = trans.transform_points(verts)
            joints = trans.transform_points(joints)
        else:  # inner translation
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            art_pose = torch.cat([axisang, art_pose], -1)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            # if art_pose.size(-1) == 45:
            verts, joints, faces = self._forward_layer(art_pose, trans, **kwargs)

        textures = torch.ones_like(verts)

        if return_mesh:
            return Meshes(verts, faces, TexturesVertex(textures)), joints
        else:
            return verts, faces, textures, joints

    def pose_to_pca(self, pose, ncomps=45):
        """
        :param pose: (N, 45)
        :return: articulated pose: (N, pca)
        """
        pose = pose - self.hand_mean
        components = self.th_selected_comps[:ncomps]  # (D, 45)
        scale = self.inv_scale[:ncomps]

        coeff = pose.mm(components.transpose(0, 1)) * scale.unsqueeze(0)
        return coeff

    def pca_to_pose(self, pca):
        """
        :param pca: (N, Dpca)
        :return: articulated pose: (N, 45)
        """
        # Remove global rot coeffs
        ncomps = pca.size(-1)
        theta = pca.mm(self.th_selected_comps[:ncomps]) + self.hand_mean
        return theta

    def cTh_transform(self, hJoints: torch.Tensor, cTh: torch.Tensor) -> Transform3d:
        """
        :param hMeshes: meshes in hand space
        :param hJoints: joints, in shape of (N, J, 3)
        :param cTh: (N, 6) intrinsic and extrinsic for a weak perspaective camera (s, x, y, rotaxisang)
        :return: cMeshes: meshes in full perspective camera space.
            se3 = geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot, ))
            mesh, j3d = hand_wrapper(se3, art_pose)
            f = 200.
            camera = PerspectiveCameras(focal_length=f,  device=device, T=torch.FloatTensor([[0, 0, .1]]).cuda())
            translate = torch.stack([tx, ty, f/s], -1)  # N, 1, 3
            mesh = mesh.update_padded(mesh.verts_padded() - j3d[:, start:start + 1] + translate)
        """
        device = hJoints.device
        if cTh.size(-1) == 7:
            f, s, tx, ty, axisang = torch.split(cTh, [1, 1, 1, 1, 3], dim=-1)
            translate = self.metric * torch.cat([tx, ty, f / s], -1)  # N, 3
            rot = Rotate(geom_utils.axis_angle_t_to_matrix(axisang, homo=False).transpose(1, 2), device=device)

            start = 5
            hJoints = rot.transform_points(hJoints)
            offset = Translate(-hJoints[:, start] + translate, device=device)

            cTh_transform = rot.compose(offset)  # R X + t
        else:
            cTh_transform = geom_utils.rt_to_transform(cTh)
        return cTh_transform

    def _forward_layer(self, pose, trans, **kwargs):
        verts, joints = self.mano_layer_right(pose, th_trans=trans, **kwargs) # in MM
        verts /= (1000 / self.metric)
        joints /= (1000 / self.metric)

        faces = self.hand_faces.repeat(verts.size(0), 1, 1)

        return verts, joints, faces

    def pose_to_transform(self, hA, include_wrist=True):
        """
        :param hA: (N, (3+)45)
        :param include_wrist:
        :return: (N, (3+)J, 4, 4)
        """
        N = hA.size(0)
        device = hA.device

        if not include_wrist:
            zeros = torch.zeros([N, 3], device=device)
            hA = torch.cat([zeros, hA], -1)

        th_pose_map, th_rot_map = th_posemap_axisang(hA)
        root_rot = th_rot_map[:, :9].view(N, 3, 3)
        th_rot_map = th_rot_map[:, 9:]

        # Full axis angle representation with root joint
        th_shapedirs = self.mano_layer_right.th_shapedirs
        th_betas = self.mano_layer_right.th_betas
        th_J_regressor = self.mano_layer_right.th_J_regressor
        th_v_template = self.mano_layer_right.th_v_template

        th_v_shaped = torch.matmul(th_shapedirs,
                                   th_betas.transpose(1, 0)).permute(
                                       2, 0, 1) + th_v_template
        th_j = torch.matmul(th_J_regressor, th_v_shaped).repeat(
            N, 1, 1)

        # Global rigid transformation
        root_j = th_j[:, 0, :].contiguous().view(N, 3, 1)  # wrist coord
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))  # homogeneousr [R, t]

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]  # (1, 16, 4, 4)
        th_results_global = th_results

        th_jtr = th_results_global
        # todo
        th_jtr = th_jtr[:, [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]]
        return th_jtr



def cvt_axisang_t_i2o(axisang, trans):
    """+correction: t_r - R_rt_r. inner to outer"""
    trans += get_offset(axisang)

    return axisang, trans


def cvt_axisang_t_o2i(axisang, trans):
    """-correction: t_r, R_rt_r. outer to inner"""
    trans -= get_offset(axisang)
    return axisang, trans


def get_offset(axisang):
    """
    :param axisang: (N, 3)
    :return: trans: (N, 3) = r_r - R_r t_r
    """
    device = axisang.device
    N = axisang.size(0)
    t_mano = torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, device=device).repeat(N, 1)
    # t_r = torch.tensor([[0.09988064, 0.01178287,  -0.01959994]], dtype=torch.float32, device=device).repeat(N, 1)
    rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
    delta = t_mano - torch.matmul(rot_r, t_mano.unsqueeze(-1)).squeeze(-1)
    return delta


def get_nTh(hand_wrapper, hA, r, inverse=False, center=None):
    """
    
    Args:
        center: (N, 3?)
        hA (N, 45 ): Description
        r (float, optional): Description
    Returns: 
        (N, 4, 4)
    """
    # add a dummy batch dim
    if center is None:
        hJoints = hand_wrapper(None, hA, mode='inner')[1]
        start = 5
        center = hJoints[:, start]  #
    device = center.device
    N = len(center)
    y = 0.08
    center = center + torch.FloatTensor([0, -y, 0]).unsqueeze(0).to(device)

    # (x - center) / r
    mat = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    mat[..., :3, :3] /= r
    mat[..., :3, 3] = -center / r

    if inverse:
        mat = geom_utils.inverse_rt(mat=mat, return_mat=True)
    return mat