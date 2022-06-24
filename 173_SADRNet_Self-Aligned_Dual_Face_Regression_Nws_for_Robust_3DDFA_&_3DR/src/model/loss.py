import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import *
import skimage.io as io
from src.dataset.uv_face import face_mask_np, face_mask_fix_rate, foreface_ind, uv_kpt_ind, uv_edges, uv_triangles

weight_mask_np = io.imread(FACE_WEIGHT_MASK_PATH).astype(float)
weight_mask_np[weight_mask_np == 255] = 256
weight_mask_np = weight_mask_np / 16
weight_mask_np[weight_mask_np == 4] = 12
weight_mask = torch.from_numpy(weight_mask_np).float()

face_mask = torch.from_numpy(face_mask_np).float()


class FaceWeightedRSE(nn.Module):
    def __init__(self):
        super(FaceWeightedRSE, self).__init__()
        self.weight_mask = weight_mask.clone()
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        if self.weight_mask.device != y_true.device:
            self.weight_mask = self.weight_mask.to(y_true.device)
            self.face_mask = self.face_mask.to(y_true.device)

        dist = torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1))
        dist = dist * self.weight_mask * self.face_mask * face_mask_fix_rate
        loss = torch.mean(dist)
        return loss


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        return F.binary_cross_entropy(y_pred, y_true)


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0)
        kernel = torch.stack([kernel, kernel, kernel])
        self.kernel = kernel
        self.face_mask = face_mask.clone()

    def forward(self, y_pred):
        if self.face_mask.device != y_pred.device:
            self.face_mask = self.face_mask.to(y_pred.device)
            self.kernel = self.kernel.to(y_pred.device)
        # foreface = y_pred * self.face_mask
        diff = F.conv2d(y_pred, self.kernel, padding=1, groups=3)
        dist = torch.norm(diff * self.face_mask, dim=1)
        loss = torch.mean(dist)
        return loss


class NME(nn.Module):
    def __init__(self, rate=1.0):
        super(NME, self).__init__()
        self.rate = rate
        self.weight_mask = weight_mask.clone()
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        if self.weight_mask.device != y_true.device:
            self.weight_mask = self.weight_mask.to(y_true.device)
            self.face_mask = self.face_mask.to(y_true.device)
        pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist) * self.rate


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_true, y_pred):
        dist = torch.mean(torch.abs(y_true - y_pred))
        return dist


class FaceRSE(nn.Module):
    def __init__(self):
        super(FaceRSE, self).__init__()
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        if self.face_mask.device != y_true.device:
            self.face_mask = self.face_mask.to(y_true.device)
        dist = torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1))
        dist = dist * (self.face_mask * face_mask_fix_rate)
        loss = torch.mean(dist)
        return loss


class KptRSE(nn.Module):
    def __init__(self):
        super(KptRSE, self).__init__()

    def forward(self, y_true, y_pred):
        gt = y_true[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        pred = y_pred[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return dist


class EdgeLengthLoss(nn.Module):
    def __init__(self, edges=uv_edges):
        super(EdgeLengthLoss, self).__init__()
        """
        edges=[ [x1,y1,x2,y2]]
        """
        self.edges = edges.astype(np.int32)
        self.edges = torch.LongTensor(self.edges)

    def forward(self, uvm_gt, uvm_out):
        if self.edges.device != uvm_out.device:
            # print(self.face.device,coord_out.device)
            self.edges = self.edges.to(uvm_out.device)
        edges = self.edges
        d_out = torch.sqrt(
            torch.sum((uvm_out[:, :, edges[:, 0], edges[:, 1]] - uvm_out[:, :, edges[:, 2], edges[:, 3]]) ** 2, dim=1,
                      keepdim=True))
        d_gt = torch.sqrt(
            torch.sum((uvm_gt[:, :, edges[:, 0], edges[:, 1]] - uvm_gt[:, :, edges[:, 2], edges[:, 3]]) ** 2, dim=1,
                      keepdim=True))
        diff = torch.abs(d_out - d_gt)
        loss = torch.mean(diff)
        return loss


class NormalVectorLoss(nn.Module):
    def __init__(self, triangles=uv_triangles):
        super(NormalVectorLoss, self).__init__()
        """
        triangles=[[x1,y1,x2,y2,x3,y3]]
        """
        self.triangles = triangles.astype(np.int32)
        self.triangles = torch.LongTensor(self.triangles)

    def forward(self, uvm_gt, uvm_out):
        if self.triangles.device != uvm_out.device:
            self.triangles = self.triangles.to(uvm_out.device)
        triangles = self.triangles
        # ab
        v1_out = uvm_out[:, :, triangles[:, 2], triangles[:, 3]] - uvm_out[:, :, triangles[:, 0], triangles[:, 1]]
        v1_out = F.normalize(v1_out, p=2, dim=1)  # make vector_length=1   L2 normalize
        # ac
        v2_out = uvm_out[:, :, triangles[:, 4], triangles[:, 5]] - uvm_out[:, :, triangles[:, 0], triangles[:, 1]]
        v2_out = F.normalize(v2_out, p=2, dim=1)
        # bc
        v3_out = uvm_out[:, :, triangles[:, 4], triangles[:, 5]] - uvm_out[:, :, triangles[:, 2], triangles[:, 3]]
        v3_out = F.normalize(v3_out, p=2, dim=1)

        # ab
        v1_gt = uvm_gt[:, :, triangles[:, 2], triangles[:, 3]] - uvm_gt[:, :, triangles[:, 0], triangles[:, 1]]
        v1_gt = F.normalize(v1_gt, p=2, dim=1)
        # ac
        v2_gt = uvm_gt[:, :, triangles[:, 4], triangles[:, 5]] - uvm_gt[:, :, triangles[:, 0], triangles[:, 1]]
        v2_gt = F.normalize(v2_gt, p=2, dim=1)

        # norm direction
        normal_gt = torch.cross(v1_gt, v2_gt, dim=1)
        normal_gt = F.normalize(normal_gt, p=2, dim=1)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 1, keepdim=True))  # cos loss
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 1, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 1, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1).mean()
        return loss


class NME2D(nn.Module):
    def __init__(self):
        super(NME2D, self).__init__()
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        if self.face_mask.device != y_true.device:
            self.face_mask = self.face_mask.to(y_true.device)
        pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred[:, :2] - gt[:, :2], dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist)


class KptNME(nn.Module):
    def __init__(self):
        super(KptNME, self).__init__()

    def forward(self, y_true, y_pred):
        gt = y_true[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        pred = y_pred[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist)


class KptNME2D(nn.Module):
    def __init__(self):
        super(KptNME2D, self).__init__()

    def forward(self, y_true, y_pred):
        gt = y_true[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        pred = y_pred[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred[:, :2] - gt[:, :2], dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist)


class FastAlignment:
    def __init__(self):
        super(FastAlignment, self).__init__()

    def __call__(self, uvm_src, uvm_dst):
        B, C, W, H = uvm_src.shape
        pts_dst = uvm_dst[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]].permute(0, 2, 1)
        pts_src = uvm_src[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]].permute(0, 2, 1)
        R, T = self.get_tform_batch(pts_src, pts_dst)
        output_uvm = uvm_src.permute(0, 2, 3, 1).reshape(B, W * H, C).matmul(R.permute(0, 2, 1)) + T.unsqueeze(1)
        output_uvm = output_uvm.reshape(B, W, H, C).permute(0, 3, 1, 2)
        return output_uvm

    def get_tform_batch(self, pts_src, pts_dst):
        # sum_dist1 = torch.sum(torch.norm(kpt_src - kpt_src[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
        # sum_dist2 = torch.sum(torch.norm(kpt_dst - kpt_dst[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
        sum_dist1 = torch.sum(torch.norm(pts_src - pts_src.mean(dim=1, keepdim=True), dim=2), dim=1).unsqueeze(
            -1).unsqueeze(-1)
        sum_dist2 = torch.sum(torch.norm(pts_dst - pts_dst.mean(dim=1, keepdim=True), dim=2), dim=1).unsqueeze(
            -1).unsqueeze(-1)
        A = pts_src * sum_dist2 / sum_dist1
        B = pts_dst
        mu_A = A.mean(dim=1, keepdim=True)
        mu_B = B.mean(dim=1, keepdim=True)
        AA = A - mu_A
        BB = B - mu_B
        H = AA.permute(0, 2, 1).matmul(BB)
        U, S, V = torch.svd(H)
        R = V.matmul(U.permute(0, 2, 1))
        t = torch.mean(B - A.matmul(R.permute(0, 2, 1)), dim=1)
        return R * sum_dist2 / sum_dist1, t


def cp(kpt_src, kpt_dst):
    sum_dist1 = np.sum(np.linalg.norm(kpt_src - kpt_src[0], axis=1))
    sum_dist2 = np.sum(np.linalg.norm(kpt_dst - kpt_dst[0], axis=1))
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.T.dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # if np.linalg.det(R) < 0:
    #     print('singular R')
    #     Vt[2, :] *= -1
    #     R = Vt.T.dot(U.T)
    t = mu_B - mu_A.dot(R.T)
    R = R * sum_dist2 / sum_dist1
    tform = np.zeros((4, 4))
    tform[0:3, 0:3] = R
    tform[0:3, 3] = t
    tform[3, 3] = 1
    return tform


class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.ICP = FastAlignment()

    def forward_torch(self, y_true, y_pred):
        aligned_pred = self.ICP(y_pred, y_true)
        outer_interocular_vec = y_true[:, :, uv_kpt_ind[36, 0], uv_kpt_ind[36, 1]] - y_true[:, :, uv_kpt_ind[45, 0],
                                                                                     uv_kpt_ind[45, 1]]
        outer_interocular_dist = torch.norm(outer_interocular_vec, dim=1, keepdim=True)
        pred = aligned_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        dist = torch.mean(torch.norm(pred - gt, p=2,dim=1), dim=1)
        dist = dist / outer_interocular_dist
        return torch.mean(dist)

    def forward(self, y_true, y_pred):
        y_true = y_true[0].cpu().permute(1, 2, 0).numpy()
        y_pred = y_pred[0].cpu().permute(1, 2, 0).numpy()

        y_pred_vertices = y_pred[face_mask_np > 0]
        y_true_vertices = y_true[face_mask_np > 0]
        # Tform, mean_dist, break_itr = icp(y_pred_vertices[0::4], y_true_vertices[0::4], max_iterations=50)

        Tform = cp(y_pred_vertices, y_true_vertices)
        # Tform, mean_dist, break_itr = icp(y_pred_vertices[0::], y_true_vertices[0::], max_iterations=5, init_pose=Tform)

        y_fit_vertices = y_pred_vertices.dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
        #
        dist = np.linalg.norm(y_fit_vertices - y_true_vertices, axis=1)

        outer_interocular_dist = y_true[uv_kpt_ind[36, 0], uv_kpt_ind[36, 1]] - y_true[
            uv_kpt_ind[45, 0], uv_kpt_ind[45, 1]]
        bbox_size = np.linalg.norm(outer_interocular_dist[0:3])

        dist = torch.from_numpy(dist)
        # loss = np.mean(dist / bbox_size)
        loss = torch.mean(dist / bbox_size)
        return loss
