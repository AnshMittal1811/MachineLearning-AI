import torch
import torch.nn.functional as F

from bilinear_sampler import bilinear_sampler1
import projective_ops as pops
# import matplotlib.pyplot as plt
import time
import subprocess

import alt_cuda_corr

import gc

import torch
import os


## code from https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
## MEM utils ##
def mem_report(file):
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        # print('Storage on %s' %(mem_type))
        # print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())
        if os.path.exists(file):
            f = open(file, "r")
            prev = float(f.readline())
            f.close()
        else:
            prev = 0
        f = open(file, "w")
        f.write(f"{max(prev, total_mem)}")
        f.close()


    LEN = 65
    # print('='*LEN)
    objects = gc.get_objects()
    # print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    # print('='*LEN)



# Inherit from Function
class DirectCorr(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fmap1, fmap2, coords):
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = alt_cuda_corr.forward(fmap1, fmap2, coords, 0)
        return corr

    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            alt_cuda_corr.backward(fmap1, fmap2, coords, grad_output, 0)

        return fmap1_grad, fmap2_grad, coords_grad


def direct_corr(fmaps, x1, ii, jj, DD):
    fmaps = fmaps.permute(0,1,3,4,2)
    fmaps1 = fmaps[:,ii] / 8.0
    fmaps2 = fmaps[:,jj] / 8.0

    batch, num, h1, w1, ch = fmaps1.shape
    fmaps1 = fmaps1.reshape(batch*num, h1, w1, ch).contiguous().float()
    fmaps2 = fmaps2.reshape(batch*num, h1, w1, ch).contiguous().float()

    x1 = x1.reshape(batch*num, h1, w1, -1, 2)
    x1 = x1.permute(0,3,1,2,4).contiguous()

    corr = DirectCorr.apply(fmaps1, fmaps2, x1)
    corr = corr.permute(0,2,3,4,1)

    return corr.reshape(batch*num*h1*w1, 1, 1, DD)


class CorrBlock:
    def __init__(self, fmaps, poses, intrinsics, ii, jj, ii_reduced, DD=128, params={}, disps_input=None, memory_file="", opt_num=1):
        # print(memory_file)
        self.memory_file = memory_file
        self.num_levels = params['num_levels'] # 5
        self.radius = params['radius'] # 5
        self.Dnear = Dnear = params['Dnear']
        self.DD = DD
        if params.__contains__("Dfar"):
            self.Dfar = params["Dfar"]
        else:
            self.Dfar = 0

        dynarange = (params["cascade"] and not disps_input is None)
        self.dynarange = dynarange
        self.inference = params["inference"]
        len_dyna = params["len_dyna"]

        device = fmaps.device
        fmaps = fmaps.float()

        batch, num_frames, ch, h1, w1 = fmaps.shape
        
            
        self.step = step = (Dnear - self.Dfar) /  float(DD)
        if not dynarange:
            disps = (Dnear - self.Dfar) * (torch.arange(DD) / float(DD)).to(device) + self.Dfar
            disps = disps.view(1, 1, DD, 1, 1)
            disps = disps.repeat(batch, opt_num, 1, h1, w1)
        else:
            self.considered_range = len_dyna
            considered_range = self.considered_range
            disps = ((torch.arange(considered_range) - considered_range // 2) * step).to(device).view(1, 1, considered_range, 1, 1)
            self.disps_origin = torch.ceil((disps_input.view(batch, opt_num, 1, h1, w1) - self.Dfar) / step) * step + self.Dfar
            disps = disps + self.disps_origin

        if not params["inference"]:
            num = ii.shape[0]
            x1 = pops.projective_transform(poses, disps, intrinsics, ii, jj, ii_reduced)
            x1 = x1[..., [0,1]].permute(0,1,3,4,2,5).contiguous()

            x1 = x1.clamp(min=-1e4, max=1e4)

            len_corr = len_dyna if dynarange else DD
            corr = direct_corr(fmaps, x1, ii, jj, len_corr)
            corr = corr.reshape(-1, 1, 1, len_corr)
            torch.cuda.empty_cache()
        else:
            num = ii.shape[0]
            segs = list(range(num + 1))
            corr_parts = []
            for j in range(len(segs) - 1):
                cur_num = segs[j + 1] - segs[j]
                x1 = pops.projective_transform(poses, disps, intrinsics, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]], ii_reduced[segs[j]:segs[j + 1]])
                # x1: B x len_ii x #disp_level x H x W x 4
                x1 = x1[..., [0,1]].permute(0,1,3,4,2,5).contiguous()

                x1 = x1.clamp(min=-1e4, max=1e4)
                torch.cuda.empty_cache()

                len_corr = len_dyna if dynarange else DD
                corr_parts.append(direct_corr(fmaps, x1, ii[segs[j]:segs[j + 1]], jj[segs[j]:segs[j + 1]], len_corr).view(batch, cur_num, h1*w1, 1, 1, len_corr))
                torch.cuda.empty_cache()                

            corr = torch.cat(corr_parts, dim=1)
            torch.cuda.empty_cache()
            corr = corr.reshape(-1, 1, 1, len_corr)
            torch.cuda.empty_cache()



        self.corr_pyramid = [ corr ]
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

            torch.cuda.empty_cache()


    def __call__(self, zinv):
        r = self.radius
        batch, num, h1, w1 = zinv.shape
        
        zinv = zinv.view(batch*num, h1, w1, 1)

        if not self.dynarange:
            coords = torch.maximum((zinv - self.Dfar) / self.step, torch.Tensor([0]).to(zinv.device))
        else:
            coords = torch.maximum((zinv - self.disps_origin.view(batch, h1, w1, 1).repeat(num, 1, 1, 1)) / self.step + self.considered_range // 2, torch.Tensor([0]).to(zinv.device))

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            if not self.inference:
                dx = torch.linspace(-r, r, 2*r+1)
                dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
                x0 = dx + coords.reshape(batch*num*h1*w1, 1, 1, 1) / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0,y0], dim=-1)
                corr = bilinear_sampler1(corr, coords_lvl)
                corr = corr.view(batch*num, h1, w1, -1)

            else:
                segs = list(range(-r, r + 1))
                corr_parts = []
                for j in range(len(segs)):
                    dx = torch.tensor([segs[j]])
                    dx = dx.view(1, 1, 1, 1).to(coords.device)
                    x0 = dx + coords.reshape(batch*num*h1*w1, 1, 1, 1) / 2**i
                    y0 = torch.zeros_like(x0)
                    coords_lvl = torch.cat([x0,y0], dim=-1)
                    torch.cuda.empty_cache()
                    sub_parts = []
                    chunk_num = 8
                    assert(batch * h1 * w1 * num % chunk_num == 0)
                    chunk_size = batch * h1 * w1 * num // chunk_num
                    for k in range(chunk_num):
                        sub_parts.append(bilinear_sampler1(corr[chunk_size * k: chunk_size * (k + 1)], coords_lvl[chunk_size * k: chunk_size * (k + 1)]))
                        torch.cuda.empty_cache()

                    corr_parts.append(torch.cat(sub_parts, 0).view(batch*num, h1, w1, -1))
                    if not self.memory_file is None:
                        mem_report(self.memory_file)
                    torch.cuda.empty_cache()
                corr = torch.cat(corr_parts, dim=-1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1).permute(0, 3, 1, 2)
        return out.reshape(batch, num, -1, h1, w1).contiguous()