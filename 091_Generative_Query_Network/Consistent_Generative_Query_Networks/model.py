import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from draw import Prior, Posterior
from encoder import Encoder
from renderer import Renderer

class JUMP(nn.Module):
    def __init__(self, nt=4, stride_to_hidden=2, nf_to_hidden=64, nf_enc=128, stride_to_obs=2, nf_to_obs=128, nf_dec=64, nf_z=3, nf_v=1):
        super(JUMP, self).__init__()
        
        # The number of DRAW steps in the network.
        self.nt = nt
        # The kernel and stride size of the conv. layer mapping the input image to the LSTM input.
        self.stride_to_hidden = stride_to_hidden
        # The number of channels in the LSTM layer.
        self.nf_to_hidden = nf_to_hidden
        # The number of channels in the conv. layer mapping the input image to the LSTM input.
        self.nf_enc = nf_enc
        # The kernel and stride size of the transposed conv. layer mapping the LSTM state to the canvas.
        self.stride_to_obs = stride_to_obs
        # The number of channels in the hidden layer between LSTM states and the canvas
        self.nf_to_obs = nf_to_obs
        # The number of channels of the conv. layer mapping the canvas state to the LSTM input.
        self.nf_dec = nf_dec
        # The number of channels in the stochastic latent in each DRAW step.
        self.nf_z = nf_z
                
        # Encoder network
        self.m_theta = Encoder(nf_v=nf_v)
            
        # DRAW
        self.prior = Prior(stride_to_hidden, nf_to_hidden, nf_enc, nf_z)
        self.posterior = Posterior(stride_to_hidden, nf_to_hidden, nf_enc, nf_z)
        
        # Renderer
        self.m_gamma = Renderer(nf_to_hidden, stride_to_obs, nf_to_obs, nf_dec, nf_z, nf_v)
        self.transconv = nn.ConvTranspose2d(nf_to_obs, 3, kernel_size=4, stride=4)
        
    # EstimateELBO
    def forward(self, v_data, f_data, pixel_var):
        B, N, C, H, W = f_data.size()
        
        M = random.randint(1, N-1)
        indices = np.random.permutation(range(N))
        context_idx, target_idx = indices[:M], indices[M:]
        v, f = v_data[:, context_idx], f_data[:, context_idx]
        v_prime, f_prime = v_data[:, target_idx], f_data[:, target_idx]
        
        r = torch.sum(self.m_theta(v, f).view(B, M, 32, H//4, W//4), dim=1)
        r_prime = torch.sum(self.m_theta(v_prime, f_prime).view(B, N-M, 32, H//4, W//4), dim=1)
        
        H_hidden, W_hidden = H//(4*self.stride_to_hidden), W//(4*self.stride_to_hidden)

        # Prior initial state
        h_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        
        # Posterior initial state
        h_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        # Renderer initial state
        h_gamma = v.new_zeros((B*(N-M), self.nf_to_hidden, H_hidden, W_hidden))
        c_gamma = v.new_zeros((B*(N-M), self.nf_to_hidden, H_hidden, W_hidden))
        canvas = v.new_zeros((B*(N-M), self.nf_to_obs, H_hidden*self.stride_to_obs, W_hidden*self.stride_to_obs))
        
        kl = 0
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)
            
            # Posterior
            h_psi, c_psi, p_psi  = self.posterior(r, r_prime, z, h_psi, c_psi)

            # Posterior sample
            z = p_psi.rsample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v_prime, canvas, h_gamma, c_gamma)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(p_psi, p_phi), dim=[1,2,3])
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N-M, C, H, W) + Normal(v.new_zeros((B, N-M, C, H, W)), np.sqrt(pixel_var)).sample()
        mse_loss = nn.MSELoss(reduction='none')
        mse = torch.sum(mse_loss(f_hat, f_prime), dim=[1,2,3,4])
        elbo = kl + mse / pixel_var
        
        return elbo, kl, mse
    
    def generate(self, v, f, v_prime):
        B, M, C, H, W = f.size()
        N = v_prime.size(1)
        
        # Scene encoder
        r = torch.sum(self.m_theta(v, f).view(B, M, 32, H//4, W//4), dim=1)
        
        H_hidden, W_hidden = H//(4*self.stride_to_hidden), W//(4*self.stride_to_hidden)

        # Prior initial state
        h_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        # Renderer initial state
        h_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        c_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        canvas = v.new_zeros((B*N, self.nf_to_obs, H_hidden*self.stride_to_obs, W_hidden*self.stride_to_obs))
                
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)

            # Prior sample
            z = p_phi.sample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v_prime, canvas, h_gamma, c_gamma)
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N, C, H, W)

        return torch.clamp(f_hat, 0, 1)
    
    def reconstruct(self, v, f):
        B, N, C, H, W = f.size()
        
        # Scene encoder
        r = torch.sum(self.m_theta(v, f).view(B, N, 32, H//4, W//4), dim=1)
        
        H_hidden, W_hidden = H//(4*self.stride_to_hidden), W//(4*self.stride_to_hidden)
        
        # Prior initial state
        h_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        # Renderer initial state
        h_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        c_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        canvas = v.new_zeros((B*N, self.nf_to_obs, H_hidden*self.stride_to_obs, W_hidden*self.stride_to_obs))
        
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)

            # Prior sample
            z = p_phi.sample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v, canvas, h_gamma, c_gamma)
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N, C, H, W)

        return torch.clamp(f_hat, 0, 1)
    