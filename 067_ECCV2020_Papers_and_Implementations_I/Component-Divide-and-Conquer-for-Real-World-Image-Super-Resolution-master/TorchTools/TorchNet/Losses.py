import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .VGG import vgg19
from .GaussianKernels import batch_kernel_param
from .modules import BatchBlur, BatchBicubicDownsampler
import pdb

def L2_loss(x1, x2, mask=1):
    """
    L2 loss, x2 can be tensor that needs grad
    :param x1:
    :param x2:
    :param mask: for augment for edges
    :return:
    """
    return torch.mean(((x1 - x2) * mask)**2)


def L1_loss(x1, x2, mask=1):
    """
    L1 loss
    :param x1:
    :param x2:
    :return:
    """
    return torch.mean(torch.abs((x1 - x2) * mask))


def C_loss(x1, x2):
    """L1 Charbonnierloss."""
    diff = torch.add(x1, -x2)
    error = torch.sqrt(diff * diff + 1e-6)
    loss = torch.sum(error)
    return loss

def GW_loss(x1, x2):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    b, c, w, h = x1.shape
    sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
    sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
    sobel_x = sobel_x.type_as(x1)
    sobel_y = sobel_y.type_as(x1)
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
    Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
    Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
    Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
    dx = torch.abs(Ix1 - Ix2)
    dy = torch.abs(Iy1 - Iy2)
#     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
    loss = (1 + 4*dx) * (1 + 4*dy) * torch.abs(x1 - x2)
    return torch.mean(loss)

def Adaptive_GW_loss(x1, x2, corner=True):
    Y_x1 = torch.mean(x1, dim=1, keepdim=True)
    Y_x2 = torch.mean(x2, dim=1, keepdim=True)
    sobel_0 = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_90 = [[-1,-2,-1],[0,0,0],[1,2,1]]
    sobel_45 = [[-2,-1,0],[-1,0,1],[0,1,2]]
    sobel_135 = [[0,-1,-2],[1,0,-1],[2,1,0]]
    
    b, c, w, h = Y_x1.shape
    sobel_0 = torch.FloatTensor(sobel_0).expand(c, 1, 3, 3)
    sobel_90 = torch.FloatTensor(sobel_90).expand(c, 1, 3, 3)
    sobel_45 = torch.FloatTensor(sobel_45).expand(c, 1, 3, 3)
    sobel_135 = torch.FloatTensor(sobel_135).expand(c, 1, 3, 3)
    sobel_0 = sobel_0.type_as(Y_x1)
    sobel_90 = sobel_90.type_as(Y_x1)
    sobel_45 = sobel_0.type_as(Y_x1)
    sobel_135 = sobel_90.type_as(Y_x1)
    
    weight_0 = nn.Parameter(data=sobel_0, requires_grad=False)
    weight_90 = nn.Parameter(data=sobel_90, requires_grad=False)
    weight_45 = nn.Parameter(data=sobel_45, requires_grad=False)
    weight_135 = nn.Parameter(data=sobel_135, requires_grad=False)
    
    I1_0 = F.conv2d(Y_x1, weight_0, stride=1, padding=1, groups=c)
    I2_0 = F.conv2d(Y_x2, weight_0, stride=1, padding=1, groups=c)
    I1_90 = F.conv2d(Y_x1, weight_90, stride=1, padding=1, groups=c)
    I2_90 = F.conv2d(Y_x2, weight_90, stride=1, padding=1, groups=c)
    I1_45 = F.conv2d(Y_x1, weight_45, stride=1, padding=1, groups=c)
    I2_45 = F.conv2d(Y_x2, weight_45, stride=1, padding=1, groups=c)
    I1_135 = F.conv2d(Y_x1, weight_135, stride=1, padding=1, groups=c)
    I2_135 = F.conv2d(Y_x2, weight_135, stride=1, padding=1, groups=c)
    d0 = torch.abs(I1_0 - I2_0)
    d90 = torch.abs(I1_90 - I2_90)
    d45 = torch.abs(I1_45 - I2_45)
    d135 = torch.abs(I1_135 - I2_135)
    
    if corner:
        d0 = d0.expand(x1.shape)
        d90 = d90.expand(x1.shape)
        d45 = d45.expand(x1.shape)
        d135 = d135.expand(x1.shape)
        loss = (1 + 4*d0) * (1 + 4*d90) *(1 + 4*d45) *(1 + 4*d135) * torch.abs(x1 - x2)
    else:
        d = torch.cat((d0, d90, d45, d135), dim=1)
        d = torch.max(d, dim=1, keepdim=True)[0]
        d = d.expand(x1.shape)
        loss = (1 + 4*d) * torch.abs(x1 - x2)
    
    return torch.mean(loss)


def get_content_loss(loss_type, nn_func=True, use_cuda=False):
    """
    content loss: [l1, l2, c]
    :param loss_type:
    :param nn_func:
    :return:
    TODO: C_loss
    """
    if loss_type == 'l2':
        loss = nn.MSELoss() if nn_func else L2_loss
    elif loss_type == 'l1':
        loss = nn.L1Loss() if nn_func else L1_loss
    elif loss_type == 'c':
        loss = C_loss
    else:
        loss = nn.MSELoss() if nn_func else L2_loss
    if use_cuda and nn_func:
        return loss.cuda()
    else:
        return loss


''' Contextual Loss '''
class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3

class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        # self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        self.cs_NHWC = self.cs_weights_before_normalization

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        r_Ts = torch.sum(Tvecs * Tvecs, 2)
        r_Is = torch.sum(Ivecs * Ivecs, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)   # d_ij

        relative_dist = cs_flow.calc_relative_distances()   # dhat_ij
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
            I_features_i = I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R
# --------------------------------------------------
#           CX loss
# --------------------------------------------------
def CX_loss(T_features, I_features, deformation=False, dis=False):
    def from_pt2tf(Tpt):
        Ttf = Tpt.permute(0, 2, 3, 1)
        return Ttf
    # N x C x H x W --> N x H x W x C
    T_features_tf = from_pt2tf(T_features)
    I_features_tf = from_pt2tf(I_features)

    # cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)
    # cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, sigma=1.0)
    cs_flow = CSFlow.create_using_L1(I_features_tf, T_features_tf, sigma=1.0)
    cs = cs_flow.cs_NHWC

    if deformation:
        deforma_sigma = 0.001
        sT = T_features_tf.shape[1:2 + 1]
        R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)
        cs *= torch.Tensor(R).unsqueeze(dim=0).cuda()

    if dis:
        CS = []
        k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
        indices = k_max_NC.cpu()
        N, C = indices.shape
        for i in range(N):
            CS.append((C - len(torch.unique(indices[i, :]))) / C)
        score = torch.FloatTensor(CS)
    else:
        k_max_NC = torch.max(torch.max(cs, dim=1)[0], dim=1)[0]
        CS = torch.mean(k_max_NC, dim=1)
        score = -torch.log(CS)
    # reduce mean over N dim
    # CX_loss = torch.mean(CX_loss)
    # return score
    return torch.mean(score)

def symetric_CX_loss(T_features, I_features):
    score = (CX_loss(T_features, I_features) + CX_loss(I_features, T_features)) / 2
    return score


class MCTHuberLoss(nn.Module):
    """
    The Huber Loss used in MCT
    """
    def __init__(self, hpyer_lambda, epsilon=0.01):
        super(MCTHuberLoss, self).__init__()
        self.epsilon = epsilon
        self.lamb = hpyer_lambda
        self.sobel = nn.Conv2d(2, 4, 3, stride=1, padding=0, groups=2)
        weight = np.array([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                  [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]],
                  [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                  [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.sobel.weight.data = torch.from_numpy(weight)
        self.sobel.bias.data = torch.from_numpy(bias)

    def forward(self, flows):
        Grad_Flow = self.sobel(flows)
        return torch.sqrt(torch.sum(Grad_Flow * Grad_Flow) + self.epsilon) * self.lamb

    def _sobel(self, flows):
        return self.sobel(flows)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1,:]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CropMarginLoss(nn.Module):
    def __init__(self, loss=nn.MSELoss, crop=5):
        super(CropMarginLoss, self).__init__()
        self.loss = loss()
        self.crop = crop

    def forward(self, input, target):
        return self.loss(input[:, :, self.crop: -self.crop, self.crop: -self.crop], target[:, :, self.crop: -self.crop, self.crop: -self.crop])


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()


    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss(nn.Module):
    """
    VGG(
    (features): Sequential(
    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (19): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )
    VGG里面有两个Sequential，feature 和 classifier
    """
    def __init__(self, vgg_path, layers='5', input='RGB', loss='l1', use_cuda=True, gpus=1):
        super(VGGLoss, self).__init__()
        ## Mean And Std
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.input = input
        vgg = vgg19()
        if vgg_path is not '':
            vgg.load_state_dict(torch.load(vgg_path))
        self.layers = [int(l) for l in layers]
        # layers_dict = [0, 4, 9, 18, 27, 36]
        # 35:Before Activation
        layers_dict = [0, 4, 9, 18, 26, 35]
        self.vgg = []
        if loss == 'l1':
            self.loss_func = nn.functional.l1_loss
        elif loss == 'l2':
            self.loss_func = nn.functional.mse_loss
        elif loss == 'cx':
            self.loss_func = CX_loss
        else:
            raise Exception('Do not support this loss.')

        i = 0
        for j in self.layers:
            Seq = nn.Sequential(*list(vgg.features.children())[layers_dict[i]:layers_dict[j]])
            if use_cuda:
                Seq.cuda()
                Seq = nn.DataParallel(Seq) if gpus > 1 else Seq
            self.vgg.append(Seq)
            i = j

    # def cuda(self, gpus=1):
    #     for Seq in self.vgg:
    #         Seq.cuda()
    #         if gpus > 1:
    #             Seq = nn.DataParallel(Seq)

    def forward(self, input, target):
        if self.input == 'RGB':
            input_R, input_G, input_B = torch.split(input, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            input_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            input_BGR = input
            target_BGR = target

        input_list = [input_BGR]
        target_list = [target_BGR]

        for Sequential in self.vgg:
            input_list.append(Sequential(input_list[-1]))
            target_list.append(Sequential(target_list[-1]))

        loss = []
        for i in range(len(self.layers)):
            loss.append(self.loss_func(input_list[i + 1], target_list[i + 1]))

        return sum(loss)


class VGGFeatureExtractor(nn.Module):
    """
    VGG(
    (features): Sequential(
    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (19): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )
    VGG里面有两个Sequential，feature 和 classifier
    """
    def __init__(self, vgg_path, layers=34, color_mode='RGB', use_cuda=True, gpus=1):
        super(VGGFeatureExtractor, self).__init__()
        # device = torch.device('gpu') if use_cuda else torch.device('cpu')
        self.color_mode = color_mode
        vgg = vgg19()
        if vgg_path is not '':
            vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(vgg.features.children())[:(layers + 1)])
        # layers_dict = [0, 4, 9, 18, 27, 36]
        # layers_dict = [0, 4, 9, 18, 26, 35]
        if gpus > 1:
            self.vgg = nn.DataParallel(self.vgg)
        if use_cuda:
            self.vgg.cuda()
        for k, v in self.vgg.named_parameters():
            v.requires_grad = False

    # def cuda(self, gpus=1):
    #     for Seq in self.vgg:
    #         Seq.cuda()

    def forward(self, input, target):
        if self.color_mode == 'RGB':
            input_R, input_G, input_B = torch.split(input, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            input_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            input_BGR = input
            target_BGR = target

        return self.vgg(input_BGR), self.vgg(target_BGR)

        # input_list = [input_BGR]
        # target_list = [target_BGR]
        #
        # for Sequential in self.vgg:
        #     input_list.append(Sequential(input_list[-1]))
        #     target_list.append(Sequential(target_list[-1]))
        #
        # loss = []
        # for i in range(len(self.layers)):
        #     loss.append(self.loss_func(input_list[i + 1], target_list[i + 1]))

        # return sum(loss)


# class GANLoss(nn.Module):
#     def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(GANLoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         if use_lsgan:
#             self.loss = nn.MSELoss()
#         else:
#             # self.loss = nn.BCELoss()
#             self.loss = nn.BCELoss()
#             self.loss = nn.BCEWithLogitsLoss()
#
#
#     def get_target_tensor(self, input, target_is_real):
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor
#
#     def __call__(self, input, target_is_real):
#         target_tensor = self.get_target_tensor(input, target_is_real)
#         return self.loss(input, target_tensor)


class ColorLoss(nn.Module):
    """
    Color loss between two blurred Image
    """
    def __init__(self, batch, ksize=15, sigma=3, scale=1, loss='l2'):
        super(ColorLoss, self).__init__()
        kernel, _ = batch_kernel_param(batch=batch, l=ksize, sigma=sigma)
        self.kernel = kernel
        self.blur = BatchBlur()
        self.scale = scale
        if scale != 1:
            self.ups = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)
        self.loss_func = get_content_loss(loss, nn_func=False)

    def cuda(self):
        self.kernel = self.kernel.cuda()
        self.blur.cuda()

    def forward(self, x1, x2):
        if self.scale != 1:
            x1 = self.ups(x1)

        B, C, W, H = x1.shape
        if C != 1:
            x1_batch = torch.split(x1, 1, dim=1)
            x2_batch = torch.split(x2, 1, dim=1)
            x1_blur = []
            x2_blur = []
            for i in range(C):
                x1_blur.append(self.blur(x1_batch[i], self.kernel))
                x2_blur.append(self.blur(x2_batch[i], self.kernel))
            x1 = torch.cat(x1_blur, dim=1)
            x2 = torch.cat(x2_blur, dim=1)

        return self.loss_func(x1, x2)


# def contextual_Loss(x, y, loss_type='l1', bw_param=1):
#     """
#     Contextual Loss Base On "The Contextual Loss for Image Transformation
# with Non-Aligned Data"
#     :param x: Feature input
#     :param y: Feature label
#     :param loss_type: [l1] TODO: l2, cosine
#     :param bw_param: Bandwidth Param
#     :return: loss
#     """
#     def create_dis_mat(x, y, loss_type='l1'):
#         B, C, H, W = x.shape
#         if loss_type == 'l1':
#             X = x.contiguous().view(B, C, H * W, 1)
#             X = X.expand(B, C, H * W, H * W)
#             Y = y.contiguous().view(B, C, H * W, 1)
#             Y = Y.expand(B, C, H * W, H * W).transpose(2, 3)
#             D = torch.mean(torch.abs(X - Y), dim=1)
#         return D
#
#     def create_m_mat(D):
#         """
#         For Calculate W, W = exp((1 - Dij / min(Di)) / h)
#         :param D:
#         :return:
#         """
#         epsilon = 1e-5
#         B, H, W = D.shape
#         D = D.contiguous().view(B * H, W)
#         M = torch.min(D, dim=1)[0].view(B, H, 1).expand(B, H, W)
#         return M + epsilon
#
#     D = create_dis_mat(x, y, loss_type)
#     M = create_m_mat(D)
#     W_mat = torch.exp((1 - D / M) / bw_param)
#
#     B, H, W = W_mat.shape
#     W_sum = W_mat.contiguous().view(B * H, W)
#     W_sum = torch.sum(W_sum, dim=1).view(B, H, 1).expand(B, H, W)
#
#     CX_mat = (W_mat / W_sum).transpose(1, 2).contiguous().view(B * W, H)
#     CX_idx = torch.max(CX_mat, dim=1)[0].view(B, W)
#     loss = - torch.log(torch.mean(CX_idx, dim=1))
#     return torch.mean(loss)


def contextual_Loss(x, y, loss_type='l1', bw_param=1):
    """
    Contextual Loss Base On "The Contextual Loss for Image Transformation"
    :param x: SR Feature
    :param y: Target Feature
    :param loss_type:
    :param bw_param: bandwidth param, 'h' in Paper
    :return:
    """
    def create_dis_mat(x, y, loss_type='l1'):
        """
        Create Distance Matrix
        """
        B, C, H, W = x.shape
        if loss_type == 'l1':
            X = x.contiguous().view(B, C, H * W, 1)
            X = X.expand(B, C, H * W, H * W)
            Y = y.contiguous().view(B, C, H * W, 1)
            Y = Y.expand(B, C, H * W, H * W).transpose(2, 3)
            D = torch.mean(torch.abs(X - Y), dim=1)
        return D

    def create_m_mat(D):
        """
        M = min_k(D_ik)
        :param D:
        :return:
        """
        epsilon = 1e-5
        B, H, W = D.shape
        D = D.contiguous().view(B * H, W)
        M = torch.min(D, dim=1)[0].view(B, H, 1).expand(B, H, W)
        return M + epsilon

    D = create_dis_mat(x, y, loss_type)
    M = create_m_mat(D)
    W_mat = torch.exp((1 - D / M) / bw_param)   # Wij = exp((1 - D`ij) / h)

    B, H, W = W_mat.shape
    W_sum = W_mat.contiguous().view(B * H, W)
    W_sum = torch.sum(W_sum, dim=1).view(B, H, 1).expand(B, H, W)   # CXij = Wij / Sum_k(Wik)

    CX_mat = (W_mat / W_sum).transpose(1, 2).contiguous().view(B * W, H)
    CX_idx = torch.max(CX_mat, dim=1)[0].view(B, W) # CX(x, y) = Sum(Max_j(CXij))
    loss = - torch.log(torch.mean(CX_idx, dim=1))   # CX_Loss(x, y) = -Log(Mean(CX(x, y)))
    return torch.mean(loss)














