import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.extractor import BasicEncoder
from modules.corr import *
import os
import cv2
import subprocess

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt
import time


def graph2edges(graph):
    verts, ii, jj, ii_reduced = [], [], [], []
    for idx, u in enumerate(graph):
        verts.append(u)
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            ii_reduced.append(idx)

    v = torch.as_tensor(verts).cuda()
    ii = torch.as_tensor(ii).cuda()
    jj = torch.as_tensor(jj).cuda()
    ii_reduced = torch.as_tensor(ii_reduced).cuda()
    return v, (ii, jj, ii_reduced)


def edge_pool(net, graph, type="maxmean"):
    _, (ii, jj, ii_reduced) = graph2edges(graph)

    net_list = []
    for u in graph:
        net_u = net[:, ii == u]
        if type == "maxmean":
            m1 = net_u.mean(dim=1)
            m2 = net_u.max(dim=1).values
            net_list.append(torch.cat([m1, m2], dim=1))
        elif type == "max":
            m2 = net_u.max(dim=1).values
            net_list.append(m2)
        elif type == "var":
            m2 = net_u.std(dim=1)
            net_list.append(m2)
        elif type == "mean":
            m1 = net_u.mean(dim=1)
            net_list.append(m1)
        elif type == "meanvar":
            m1 = net_u.mean(dim=1)
            m2 = net_u.std(dim=1)
            net_list.append(torch.cat([m1, m2], dim=1))
    return torch.stack(net_list, dim=1)


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128, params={}):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        kernel_z = params['kernel_z']  # 3
        kernel_r = params['kernel_r']  # 3
        kernel_q = params['kernel_q']  # 3
        self.convz = nn.Conv2d(h_planes + i_planes, h_planes, kernel_z, padding=kernel_z // 2)
        self.convr = nn.Conv2d(h_planes + i_planes, h_planes, kernel_r, padding=kernel_r // 2)
        self.convq = nn.Conv2d(h_planes + i_planes, h_planes, kernel_q, padding=kernel_q // 2)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)
        z = torch.sigmoid(self.convz(net_inp))
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)))
        net = (1 - z) * net + z * q
        return net


class UpdateBlock(nn.Module):
    def __init__(self, has_delta2, params):
        super(UpdateBlock, self).__init__()

        kernel_z = params['kernel_z']  # 3
        kernel_r = params['kernel_r']  # 3
        kernel_q = params['kernel_q']  # 3
        kernel_corr = params['kernel_corr']  # 3
        dim0_corr = params['dim0_corr']  # 128
        dim1_corr = params['dim1_corr']  # 128
        dim_net = params['dim_net']  # 128
        dim_inp = params['dim_inp']  # 128
        dim0_delta = params['dim0_delta']  # 256
        kernel0_delta = params['kernel0_delta']  # 3
        kernel1_delta = params['kernel1_delta']  # 3
        num_levels = params['num_levels']  # 5
        radius = params['radius']  # 5
        corr_len = params["corr_len"]
        self.s_disp_enc = params['s_disp_enc']  # 7

        has_delta2 = self.has_delta2 = has_delta2

        cor_planes = corr_len * num_levels * (2 * radius + 1)

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, dim0_corr, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim0_corr, dim1_corr, kernel_corr, padding=kernel_corr // 2),
            nn.ReLU(inplace=True))

        self.delta = nn.Sequential(
            nn.Conv2d(dim_net, dim0_delta, kernel0_delta, padding=kernel0_delta // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim0_delta, 1, kernel1_delta, padding=kernel1_delta // 2))

        """In cascaded model, stage 2 has different delta net"""
        if has_delta2:
            self.delta2 = nn.Sequential(
                nn.Conv2d(dim_net, dim0_delta, kernel0_delta, padding=kernel0_delta // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim0_delta, 1, kernel1_delta, padding=kernel1_delta // 2))

        i_planes = dim_inp + dim1_corr + self.s_disp_enc ** 2
        h_planes = dim_net

        self.gru = ConvGRU(h_planes, i_planes, params)

    def disp_encoder(self, disp):
        batch, _, ht, wd = disp.shape
        disp7x7 = F.unfold(disp, [self.s_disp_enc, self.s_disp_enc], padding=self.s_disp_enc // 2)
        disp7x7 = disp7x7.view(batch, self.s_disp_enc ** 2, ht, wd)
        disp1x1 = disp.view(batch, 1, ht, wd)

        return disp7x7 - disp1x1

    def forward(self, net, inp, disp, corr, stage=0):

        batch, num, ch, ht, wd = net.shape
        inp_shape = (batch * num, -1, ht, wd)
        out_shape = (batch, num, -1, ht, wd)

        net = net.view(*inp_shape)
        inp = inp.view(*inp_shape)
        corr = corr.view(*inp_shape)
        disp = disp.view(*inp_shape)

        disp = 100 * self.disp_encoder(disp)
        corr = self.corr_encoder(corr)

        assert (not torch.isnan(net).any())
        assert (not torch.isnan(inp).any())
        assert (not torch.isnan(disp).any())
        assert (not torch.isnan(corr).any())

        net = self.gru(net, inp, disp, corr)

        if stage == 0:
            delta = .01 * self.delta(net)
        else:
            delta = .01 * self.delta2(net)

        net = net.view(*out_shape)
        delta = delta.view(*out_shape)
        delta = delta.squeeze(2)
        return net, delta


class RAFT(nn.Module):
    def __init__(self, **params):
        super(RAFT, self).__init__()

        try:
            self.memory_file = params["memory_file"]
        except:
            self.memory_file = None

        self.DD = params['DD']
        self.Dnear = params['Dnear']
        self.num_iters = params['num_iters']  # 16
        self.dim_fmap = dim_fmap = params['dim_fmap']  # 128
        self.dim_net = params['dim_net']  # 128
        self.dim_inp = params['dim_inp']  # 128
        self.params = params
        self.DD_fine = params["DD_fine"]
        self.cascade = params["cascade"]
        self.cascade_v2 = params["cascade_v2"]

        self.fnet = BasicEncoder(output_dim=dim_fmap, norm_fn='instance', HR=params["HR"])
        self.cnet = BasicEncoder(output_dim=self.dim_net + self.dim_inp, norm_fn='none', HR=params["HR"])

        self.update_block = UpdateBlock(has_delta2=self.cascade_v2, params=params)

    def forward(self, images, poses, intrinsics, graph):
        """ depth inference """

        intrinsics = intrinsics.clone()
        HR = self.params["HR"]
        cascade = self.params["cascade"]
        factor = 8 if not HR else 4

        intrinsics[:, :, 0] /= factor
        intrinsics[:, :, 1] /= factor

        images *= 2 / 255.
        images -= 1

        batch, num, ch, ht, wd = images.shape
        v, (ii, jj, ii_reduced) = graph2edges(graph)

        disp = torch.zeros(batch, v.shape[0], ht // factor, wd // factor)
        disp = disp.to(images.device).float()

        h1, w1 = ht // factor, wd // factor

        with autocast(enabled=True):
            net_inp = self.cnet(images[:, v])

            net, inp = net_inp.split([self.dim_net, self.dim_inp], dim=2)

            net = torch.tanh(net)
            inp = torch.relu(inp)

            fmaps = self.fnet(images)

            with autocast(enabled=False):
                corr_fn = CorrBlock(fmaps, poses, intrinsics, ii, jj, ii_reduced, DD=self.DD,
                                    params=self.params, memory_file=self.memory_file,
                                    opt_num=v.shape[0])  ## DD separate from other params for reasons

            if self.params["inference"]:
                if not cascade: del fmaps
                del images

            torch.cuda.empty_cache()
            predictions = []

            iter1 = self.num_iters if not cascade else self.params["num_iters1"]
            for itr in range(iter1):
                tic = time.time()
                disp = disp.detach()
                with autocast(enabled=False):
                    corr = corr_fn(disp[:, ii_reduced])
                    corr = edge_pool(corr, graph, self.params['pooltype'])

                net, delta = self.update_block(net, inp, disp, corr)
                disp = disp + delta.float()
                prediction = disp
                predictions.append(prediction)

            if cascade:

                with autocast(enabled=False):
                    corr_fn = CorrBlock(fmaps, poses, intrinsics, ii, jj, ii_reduced, DD=self.DD_fine,
                                        params=self.params, disps_input=disp, memory_file=self.memory_file)
                    torch.cuda.empty_cache()

                del fmaps
                torch.cuda.empty_cache()

                for itr in range(self.params["num_iters2"]):
                    disp = disp.detach()
                    with autocast(enabled=False):
                        corr = corr_fn(disp[:, ii_reduced])
                        corr = edge_pool(corr, graph, self.params['pooltype'])
                    net, delta = self.update_block(net, inp, disp, corr, stage=self.cascade_v2)
                    disp = disp + delta.float()
                    prediction = disp
                    predictions.append(prediction)
                    torch.cuda.empty_cache()

        return predictions
