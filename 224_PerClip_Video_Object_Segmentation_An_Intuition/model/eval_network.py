"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.intra_clip_refinement import build_Intra_Clip_Refinement

class PCVOS(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.refine_clip = opt['refine_clip']
        self.memory_read = opt['memory_read']

        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder() 

        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        if self.refine_clip == 'ICR':
            self.input_proj = KeyProjection(1024, keydim=opt['hidden_dim'])
            self.refine_transformer = build_Intra_Clip_Refinement(opt)

        self.decoder = Decoder(input_dim=1024)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    ## per-clip inference
    def segment_with_query_per_clip(self, mem_bank, qf16, qf8, qf4, qk16, qv16, num_frames): 
        k = mem_bank.num_objects

        if self.memory_read == 'PMM':
            B,C,nH,W = qk16.shape
            qk16 = qk16.view(B,C,num_frames,nH//num_frames,W)
            readout_mem = mem_bank.match_memory_PMM(qk16)
        else:
            readout_mem = mem_bank.match_memory(qk16)
        
        B,C,nH,W = readout_mem.shape
        readout_mem = readout_mem.view(B,C,num_frames,nH//num_frames,W).transpose(1,2)
        
        qv16 = qv16.expand(k, -1, -1, -1, -1)
        if self.refine_clip == 'ICR':
            val_in = readout_mem.transpose(1,2)

            b, t = qf16.shape[:2]
            key_in = self.input_proj(qf16.flatten(start_dim=0, end_dim=1))
            key_in = key_in.view(b, t, *key_in.shape[1:]).transpose(1,2)
            key_in = key_in.expand(k, -1, -1, -1, -1)
            # apply intra-clip refinement
            val_out = self.refine_transformer(key_in, val_in)
            # concat
            qv16 = torch.cat([val_out.transpose(1,2), qv16], dim=2).flatten(0,1)
        else:
            qv16 = torch.cat([readout_mem, qv16], 2).flatten(0,1)
        qf8 = qf8.expand(k, -1, -1, -1, -1).flatten(0,1)
        qf4 = qf4.expand(k, -1, -1, -1, -1).flatten(0,1)
        
        out = torch.sigmoid(self.decoder(qv16, qf8, qf4))

        return out.view(k,num_frames,1,*out.shape[-2:])