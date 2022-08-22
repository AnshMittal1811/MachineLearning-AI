import torch
import torch.nn as nn
from collections import OrderedDict
from model import hyperlayers

import logging
logger = logging.getLogger(__package__)

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


class MLPNet(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=True, stage="first"):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch_position = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.hidden_dim = W
        self.stage = stage
        
        self.linear0 = self.build_base_layer(self.input_ch_position, self.hidden_dim)

        # Block 1
        self.z_linear1 = nn.Linear(64, self.hidden_dim)
        self.linear1 = self.build_base_layer(self.hidden_dim, self.hidden_dim)
        self.linear2 = self.build_base_layer(self.hidden_dim, self.hidden_dim)
        
        # Block 2
        self.z_linear2 = nn.Linear(64, self.hidden_dim)
        self.linear3 = self.build_base_layer(self.hidden_dim, self.hidden_dim)
        self.linear4 = self.build_base_layer(self.hidden_dim, self.hidden_dim)

        # For density branch
        self.sigma_layer0 = self.build_base_layer(self.hidden_dim, self.hidden_dim // 2)
        self.sigma_layer1 = nn.Linear(self.hidden_dim // 2, 1)
 
        if stage == "first":
            # Block 3
            self.base_remap_layers = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.z_linear3 = nn.Linear(64, self.hidden_dim)
            self.rgb_layer0 = self.build_base_layer(self.hidden_dim + self.input_ch_viewdirs, self.hidden_dim)
            self.rgb_layer1 = self.build_base_layer(self.hidden_dim, self.hidden_dim)

            # For RGB branch
            self.rgb_layer2 = self.build_base_layer(self.hidden_dim, self.hidden_dim // 2)
            self.rgb_layer3 = nn.Linear(self.hidden_dim // 2, 3)
            self.rgb_sigmoid = nn.Sigmoid()
        else: 
            # Use hyper-network to represent Block 3 and RGB branch
            self.hyper_rgb = hyperlayers.HyperFC(in_ch_pos=self.input_ch_position,
                                                 in_ch_view=self.input_ch_viewdirs,
                                                 out_ch=3)
            self.rgb_sigmoid = nn.Sigmoid()

    def build_base_layer(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def forward(self, world_pts, target_viewdirs, latent, style_latent):
        '''
        :return [..., 4]
        '''
        x0 = self.linear0(world_pts)

        # Block 1
        z1 = self.z_linear1(latent)
        x0 = x0 + z1
        x1 = self.linear1(x0)
        x2 = self.linear2(x1)
        x2 = x2 + x0

        # Block2
        z2 = self.z_linear2(latent)
        x2 = x2 + z2
        x3 = self.linear3(x2)
        x4 = self.linear4(x3)
        x4 = x4 + x2

        # For density branch
        sigma1 = self.sigma_layer0(x4)
        sigma2 = self.sigma_layer1(sigma1)
        sigma = torch.abs(sigma2).squeeze(-1)
        
        # Block 3
        if self.stage == "first":
            base_remap = self.base_remap_layers(x4)
            z3 = self.z_linear3(latent)
            base_remap = base_remap + z3
            rgb0 = self.rgb_layer0(torch.cat((base_remap, target_viewdirs), dim=-1))
            rgb1 = self.rgb_layer1(rgb0)

	    # For RGB branch
            rgb2 = self.rgb_layer2(rgb1)
            rgb3 = self.rgb_layer3(rgb2)
            rgb = self.rgb_sigmoid(rgb3) 
        else:
            hyper_rgb = self.hyper_rgb(style_latent)#([1, 512])->5net
            base_remap = hyper_rgb[0](x4)#([5427, 64, 128])->([5427, 64, 128])
            rgb0 = hyper_rgb[1](torch.cat((base_remap, target_viewdirs), dim=-1))#([5427, 64, 128])+([5427, 64, 27])->([5427, 64, 128])
            rgb1 = hyper_rgb[2](rgb0)#([5427, 64, 128])->([5427, 64, 128])

	    # For RGB branch
            rgb2 = hyper_rgb[3](rgb1)#([5427, 64, 128])->([5427, 64, 64])
            rgb3 = hyper_rgb[4](rgb2)#([5427, 64, 64])->([5427, 64, 3])
            rgb = self.rgb_sigmoid(rgb3)#([5427, 64, 3])

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma)])

        return ret

class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             stage=args.stage)
        
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             stage=args.stage)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, latent, style_latent):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm      # [..., 3]
        dots_sh = list(ray_d.shape[:-1])
        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        
        input_pts = self.fg_embedder_position(fg_pts)
        input_viewdirs = self.fg_embedder_viewdir(fg_viewdirs)
        fg_raw = self.fg_net(input_pts, input_viewdirs, latent, style_latent)#([5427, 64, 3])
        
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]), dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T     # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]

        input_pts = torch.flip(self.bg_embedder_position(bg_pts), dims=[-2,])
        input_viewdirs = torch.flip(self.bg_embedder_viewdir(bg_viewdirs), dims=[-2,])
        # near_depth: physical far; far_depth: physical near
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input_pts, input_viewdirs, latent, style_latent)
        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = fg_rgb_map + bg_rgb_map

        ret = OrderedDict([('rgb', rgb_map),            # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),      # below are for logging
                           ('fg_depth', fg_depth_map),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map),
                           ('bg_lambda', bg_lambda)])
        return ret

if __name__ == "__main__":
    model = MLPNet()
    print(model)
