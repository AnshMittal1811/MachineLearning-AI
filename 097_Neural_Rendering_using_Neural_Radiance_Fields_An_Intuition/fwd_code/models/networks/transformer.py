import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, q, k, v, mask=None, layer_norm=True):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual
        if layer_norm:
            q = self.layer_norm(q)
        return q, attn


class Image_Fusion_Transformer(nn.Module):

    def __init__(self, opt):

        super().__init__()

        self.model = MultiHeadAttention(opt.atten_n_head, opt.decode_in_dim, opt.atten_k_dim, opt.atten_v_dim)
        self.token =  nn.Parameter(torch.randn(1, 1, opt.atten_v_dim))
        self.opt = opt

    def forward(self, gen_fs_collect, pos_end=None):
        num_in, num_out, C, H, W = gen_fs_collect.shape
        gen_fs_collect = gen_fs_collect.permute(1, 3, 4, 0, 2).contiguous().view(-1, num_in, C)
        if pos_end is None:
            gen_fs_collect, _ = self.model(self.token.expand(H*W*num_out, 1, C), gen_fs_collect, gen_fs_collect, layer_norm=self.opt.atten_norm) # N, 1, C
        else:
            pos_end = pos_end.permute(1, 3, 4, 0, 2).contiguous().view(-1, num_in, C)
            gen_fs_collect, _ = self.model(self.token.expand(H*W*num_out, 1, C), gen_fs_collect + pos_end, gen_fs_collect, layer_norm=self.opt.atten_norm) # N, 1, C
        gen_fs_collect = gen_fs_collect.view(num_out, H, W, C) #n um_out, H, W, C

        gen_fs_collect = gen_fs_collect.contiguous().permute(0, 3, 1, 2) # (num_out, C, H, W)

        return gen_fs_collect
