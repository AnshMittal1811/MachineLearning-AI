import math

import torch.nn.functional as F
import numpy as np
import geometry
import torchvision
import util
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
from torch.nn.init import _calculate_correct_fan

from pdb import set_trace as pdb

import torch
from torch import nn

import conv2d_gradfix


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

def sal_init(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)

def sal_init_last_layer(m):
    if hasattr(m, 'weight'):
        val = np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in'))
        with torch.no_grad():
            m.weight.fill_(val)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class DotProductAttention(nn.Module):
    def __init__(self, attended_to_dim, attending_dim, num_heads=1):
        super().__init__()

        self.to_q = nn.ModuleList([BatchLinear(attending_dim, attending_dim) for _ in range(num_heads)])
        self.to_k = nn.ModuleList([BatchLinear(attended_to_dim, attending_dim) for _ in range(num_heads)])

        self.scale = attending_dim ** -0.5
        self.eps = 1e-9

    def forward(self, key, value, query):
        '''attending shape: (b, attending_dim)
        attended_to shape: (b, slots, attended_to_dim)'''
        for to_q, to_k in zip(self.to_q, self.to_k):
            q, k = to_q(query), to_k(key)

            print(q.shape, k.shape)
            dots = torch.einsum('...d,...id->...i', q, k) * self.scale
            attn = dots.softmax(dim=-1) + self.eps
            print(attn.shape, q.shape, k.shape, value.shape)
            output = torch.einsum('...i,...id->...d', attn, value)

        return output


class SelfAttention(nn.Module):
    def __init__(self, in_dim, output_dim, num_heads=1):
        super().__init__()

        self.norm_input = nn.LayerNorm(in_dim)

        self.to_q = nn.ModuleList([BatchLinear(in_dim, output_dim) for _ in range(num_heads)])
        self.to_k = nn.ModuleList([BatchLinear(in_dim, output_dim) for _ in range(num_heads)])
        self.to_v = nn.ModuleList([BatchLinear(in_dim, output_dim) for _ in range(num_heads)])

        self.result_proj = BatchLinear(output_dim*num_heads, output_dim)

        self.scale = output_dim ** -0.5
        self.eps = 1e-9

    def forward(self, input):
        input = self.norm_input(input)

        head_results = []
        for to_q, to_k, to_v in zip(self.to_q, self.to_k, self.to_v):
            q, k, v = to_q(input), to_k(input), to_v(input)

            dots = torch.einsum('b...jd,b...id->b...ji', q, k) * self.scale
            attn = dots.softmax(dim=-1) + self.eps
            output = torch.einsum('b...ji,b...id->b...jd', attn, v)
            head_results.append(output)

        output = self.result_proj(torch.cat(head_results, dim=-1))
        return output


class FCLayer(MetaModule):
    def __init__(self, in_features, out_features, nonlinearity='relu', norm=None):
        super().__init__()
        self.net = [BatchLinear(in_features, out_features)]

        if norm == 'layernorm':
            self.net.append(nn.LayerNorm([out_features], elementwise_affine=True),)
        elif norm == 'layernorm_na':
            self.net.append(nn.LayerNorm([out_features], elementwise_affine=False),)

        if nonlinearity == 'relu':
            self.net.append(nn.ReLU(inplace=True))
        elif nonlinearity == 'leaky_relu':
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, 'net'))


class FCBlock(MetaModule):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False,
                 norm=None,
                 activation='relu',
                 nonlinearity='relu'):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch, nonlinearity=nonlinearity, norm=norm))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch, nonlinearity=nonlinearity, norm=norm))

        if outermost_linear:
            self.net.append(BatchLinear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features, nonlinearity=nonlinearity, norm=norm))

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, 'net'))


class PosEncoding(MetaModule):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = BatchLinear(in_features, out_features)
        self.linear.apply(first_layer_sine_init)
        self.omega_0 = omega_0

    def forward(self, input, params=None):
        if params is None:
            params = dict(self.meta_named_parameters())

        intermed = self.omega_0 * self.linear(input, params=self.get_subdict(params, 'linear'))
        return torch.sin(intermed)


class PosEncodingFC(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, norm=None, nonlinearity='relu'):
        super().__init__()
        self.net = []
        self.net.append(PosEncoding(in_features=in_features, out_features=hidden_features, omega_0=first_omega_0))

        for i in range(hidden_layers):
            if not i:
                in_feats = hidden_features
            else:
                in_feats = hidden_features
            self.net.append(FCLayer(in_features=in_feats, out_features=hidden_features, norm=norm, nonlinearity=nonlinearity))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)
            nn.init.xavier_normal_(final_linear.weight)
            self.net.append(final_linear)
        else:
            self.net.append(FCLayer(hidden_features, out_features, norm=norm))

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, params=None):
        x = coords

        for i, layer in enumerate(self.net):
            x = layer(x, params=self.get_subdict(params, f'net.{i}'))
        return x


class Raymarcher(nn.Module):
    def __init__(self,
                 num_feature_channels,
                 raymarch_steps,
                 use_lstm=True,
                 project_on_surface=False):
        super().__init__()

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps
        self.use_lstm = use_lstm
        self.project_on_surface = project_on_surface

        if self.use_lstm:
            hidden_size = 16
            self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                    hidden_size=hidden_size)

            self.lstm.apply(init_recurrent_weights)
            lstm_forget_gate_init(self.lstm)

            self.out_layer = nn.Linear(hidden_size, 1)
        else:
            self.sdf = BatchLinear(self.n_feature_channels, 1)
            nn.init.kaiming_normal_(self.sdf.weight, a=0.0, nonlinearity='relu', mode='fan_in')
            self.sdf.bias.data = torch.ones_like(self.sdf.bias) * 0.01
            self.sdf.weight.data *= 1e-2

        self.counter = 0

    def forward(self, cam2world, phi, uv, intrinsics):

        batch_size, num_samples, _ = uv.shape
        log = list()

        ray_dirs = geometry.get_ray_directions(uv,
                                               cam2world=cam2world,
                                               intrinsics=intrinsics)

        initial_depth = torch.zeros((batch_size, num_samples, 1)).normal_(mean=0.05, std=5e-4).cuda()
        init_world_coords = geometry.world_from_xy_depth(uv,
                                                         initial_depth,
                                                         intrinsics=intrinsics,
                                                         cam2world=cam2world)

        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]

        for step in range(self.steps):
            v = phi(world_coords[-1])

            if self.use_lstm:
                state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])
                if state[0].requires_grad:
                    state[0].register_hook(lambda x: torch.clamp(x, -10., 10.))

                signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
                states.append(state)
            else:
                signed_distance = self.sdf(v)
                if signed_distance.requires_grad:
                    signed_distance.register_hook(lambda x: torch.clamp(x, -2., 2.))

            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            world_coords.append(new_world_coords)

            depth = geometry.depth_from_world(world_coords[-1], cam2world)

            if self.training:
                print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
                      (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)

        return {'coords':world_coords[-1], 'depth':depths[-1], 'all_depth':torch.stack(depths, dim=0), 'log':log}


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class MFN(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 nonlinearity='gabor'):
        super().__init__()

        if nonlinearity=='sine':
            nl_layer = SineLayer
        elif nonlinearity=='gabor':
            nl_layer = GaborLayer

        self.linear_layers = []
        self.nonlinear_layers = []

        self.in_layer = nl_layer(in_features, hidden_features, bias=True)

        self.linear_layers.append(BatchLinear(hidden_features, hidden_features))
        self.nonlinear_layers.append(nl_layer(in_features, hidden_features, bias=True, omega_0=1.))

        for i in range(hidden_layers):
            self.linear_layers.append(BatchLinear(hidden_features, hidden_features))
            self.nonlinear_layers.append(nl_layer(in_features, hidden_features, bias=True, omega_0=1.))

        self.linear_layers.append(BatchLinear(hidden_features, out_features))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.nonlinear_layers = nn.ModuleList(self.nonlinear_layers)

    def forward(self, x, params=None):
        z = self.in_layer(x)

        for i, (ln, nl) in enumerate(zip(self.linear_layers, self.nonlinear_layers)):
            z = ln(z) * nl(x)

        return self.linear_layers[-1](z)


class GaborLayer(MetaModule):
    def __init__(self, in_features, out_features, bias=None, omega_0=None, is_first=False):
        super().__init__()
        self.sine_layer = SineLayer(in_features=in_features, out_features=out_features)
        self.scale = nn.Parameter(torch.rand(out_features))
        self.mean = nn.Parameter(torch.zeros(out_features, in_features).uniform_(-1, 1))

    def forward(self, input, params=None):
        dist_to_center = ((input.unsqueeze(-2) - self.mean[None, None]).norm(dim=-1)**2)
        sine = self.sine_layer(input)
        exp = torch.exp(-torch.abs(self.scale)/2 * dist_to_center)
        return sine * exp


class SineLayer(MetaModule):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = float(omega_0)

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward_with_film(self, input, gamma, beta):
        intermed = self.linear(input)
        return torch.sin(gamma * self.omega_0 * intermed + beta)

    def forward(self, input, params=None):
        intermed = self.linear(input, params=self.get_subdict(params, 'linear'))
        return torch.sin(self.omega_0 * intermed)


class Siren(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., special_first=True):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        layer = SineLayer

        self.net = []
        self.net.append(layer(in_features, hidden_features,
                              is_first=special_first, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(layer(hidden_features, hidden_features,
                                  is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30.,
                                             np.sqrt(6 / hidden_features) / 30.)
            self.net.append(final_linear)
        else:
            self.net.append(layer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, params=None):
        x = coords

        for i, layer in enumerate(self.net):
            x = layer(x, params=self.get_subdict(params, f'net.{i}'))

        return x

    def forward_with_film(self, coords, film):
        x = coords

        for i, (layer, layer_film) in enumerate(zip(self.net, film)):
            if i < len(self.net) - 1:
                x = layer.forward_with_film(x, layer_film['gamma'], layer_film['beta'])
            else:
                x = layer.forward(x)

        return x


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # self.norm_0 = nn.LayerNorm([size_in], elementwise_affine=False)
        # self.norm_1 = nn.LayerNorm([size_h], elementwise_affine=False)
        self.norm_0 = nn.Sequential()
        self.norm_1 = nn.Sequential()

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU(inplace=True)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        net = self.fc_0(self.activation(self.norm_0(x)))
        dx = self.fc_1(self.activation(self.norm_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx

# Taken from Koven's UORF which is adapted from user lucidrains

class SlotAttentionFG(nn.Module):
    def __init__(self, num_slots, in_dim=128, slot_dim=64,
             learned_emb=False, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()

        self.learned_emb = learned_emb

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        """
        if self.learned_emb:
            self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))
            nn.init.xavier_uniform_(self.slots)
        else:
            zzz
            self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
            nn.init.xavier_uniform_(self.slots_logsigma)
        """
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q    = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat   = nn.LayerNorm(in_dim)
        self.slot_dim    = slot_dim

    def forward(self, feat, num_slots=None, slot=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        """
        if self.learned_emb:
            slot = self.slots.expand(B,-1,-1) if slot is None else slot
            #slot_bg,slot_fg = slots[:,:1],slots[:,1:]
        elif slot is None:
            zzz
            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slot= mu + sigma * torch.randn_like(mu)
        """
        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_logsigma.exp().expand(B, K, -1)
        slot= mu + sigma * torch.randn_like(mu)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for i in range(self.iters):
            slot_prev= slot
            q= self.to_q(slot)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)  # Bx1xN
            updates= torch.einsum('bjd,bij->bid', v, attn_weights)

            slot = self.gru(
                updates.reshape(-1, self.slot_dim),
                slot_prev.reshape(-1, self.slot_dim)
            )
            slot= slot.reshape(B, -1, self.slot_dim)
            slot= slot+ self.to_res(slot)

        return slot,attn


class SlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim=128, bg_slot_dim=64, fg_slot_dim=64,
                             max_slot_dim=64,iters=3, eps=1e-8, hidden_dim=128,
                        learned_emb=False
                ):
        super().__init__()

        self.learned_emb = learned_emb 

        slot_dim=max_slot_dim

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5 #note that if we anneal choose the anneal dim here

        self.slots_mu = nn.Parameter(torch.randn(1, 1, fg_slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, fg_slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, bg_slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, bg_slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma_bg)
        """
        self.slots = nn.Parameter(torch.randn(1, num_slots, fg_slot_dim))
        nn.init.xavier_uniform_(self.slots)
        """

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q    = nn.Sequential(nn.LayerNorm(fg_slot_dim), nn.Linear(fg_slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(bg_slot_dim), nn.Linear(bg_slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, fg_slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, bg_slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(fg_slot_dim),
            nn.Linear(fg_slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, fg_slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(bg_slot_dim),
            nn.Linear(bg_slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bg_slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = max_slot_dim
        self.bg_slot_dim = bg_slot_dim
        self.fg_slot_dim = fg_slot_dim

    def forward(self, feat, anneal=1, num_slots=None,slot=None,iters=3):
        torch.manual_seed(42)
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        if slot is None:
            mu_bg = self.slots_mu_bg.expand(B, 1, -1)
            sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
            slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
            mu = self.slots_mu.expand(B, K-1, -1)
            sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
            slot_fg = mu + sigma * torch.randn_like(mu)
        else:
            slot_bg,slot_fg = slot[:,:1,:self.bg_slot_dim],slot[:,1:]
        """
        slots = self.slots.expand(B,-1,-1) if slot is None else slot
        slot_bg,slot_fg = slots[:,:1],slots[:,1:]
        """

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for i in range(iters):

            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg

            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale

            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)
            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.bg_slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.bg_slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.fg_slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.fg_slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)
        if self.bg_slot_dim!=self.fg_slot_dim:
            slot_bg = torch.cat((slot_bg,torch.ones_like(
                        slot_fg[:,:1,:self.fg_slot_dim-self.bg_slot_dim])),-1)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots,attn

# Koven's GAN discriminator

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        stride=1,
        padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False,
            bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


"""
class Discriminator(nn.Module):
    def __init__(self, size, ndf, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf*2,
            8: ndf*2,
            16: ndf,
            32: ndf,
            64: ndf//2,
            128: ndf//2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc,ndf=3,64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

