import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.embedder import get_embedder

import os
import imageio


def fibonacci_sphere(samples=1):
    '''
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 num_lgt_sgs=32,
                 upper_hemi=False,
                 specular_albedo=0.02,
                 latent_dim=32):
        super().__init__()

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)

        self.numLgtSGs = num_lgt_sgs
        self.envmap = None

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)
        ############## spatially-varying BRDF ############
        
        print('BRDF encoder network size: ', brdf_encoder_dims)
        print('BRDF decoder network size: ', brdf_decoder_dims)

        brdf_encoder_layer = []
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, 4))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)

        ############## fresnel ############
        spec = torch.zeros([1, 1])
        spec[:] = specular_albedo
        self.specular_reflectance = nn.Parameter(spec, requires_grad=False)
        
        ################### light SGs ####################
        print('Number of Light SG: ', self.numLgtSGs)

        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
        self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
        # init envmap energy
        energy = compute_energy(self.lgtSGs.data)
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
        self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
        self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)
        
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

    def forward(self, points):
        if self.brdf_embed_fn is not None:
            points = self.brdf_embed_fn(points)

        brdf_lc = torch.sigmoid(self.brdf_encoder_layer(points))
        brdf = torch.sigmoid(self.brdf_decoder_layer(brdf_lc))
        roughness = brdf[..., 3:] * 0.9 + 0.09
        diffuse_albedo = brdf[..., :3]

        rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * 0.01
        random_xi_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_lc))
        random_xi_roughness = random_xi_brdf[..., 3:] * 0.9 + 0.09
        random_xi_diffuse = random_xi_brdf[..., :3]

        lgtSGs = self.lgtSGs
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        specular_reflectance = self.specular_reflectance
        self.specular_reflectance.requires_grad = False

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectance),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('random_xi_roughness', random_xi_roughness),
            ('random_xi_diffuse_albedo', random_xi_diffuse),
        ])
        return ret

    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        # limit lobes to upper hemisphere
        if self.upper_hemi:
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def load_light(self, path):
        sg_path = os.path.join(path, 'sg_128.npy')
        device = self.lgtSGs.data.device
        load_sgs = torch.from_numpy(np.load(sg_path)).to(device)
        self.lgtSGs.data = load_sgs

        energy = compute_energy(self.lgtSGs.data)
        print('loaded envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        envmap_path = path + '.exr'
        envmap = np.float32(imageio.imread(envmap_path)[:, :, :3])
        self.envmap = torch.from_numpy(envmap).to(device)
