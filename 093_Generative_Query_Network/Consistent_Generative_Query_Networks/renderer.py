import torch
from torch import nn
from conv_lstm import Conv2dLSTMCell

class Renderer(nn.Module):
    def __init__(self, nf_to_hidden, stride_to_obs, nf_to_obs, nf_dec, nf_z, nf_v):
        super(Renderer, self).__init__()
        self.conv = nn.Conv2d(nf_to_obs, nf_dec, kernel_size=stride_to_obs, stride=stride_to_obs)
        self.lstm = Conv2dLSTMCell(nf_z+nf_v+nf_dec, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.transconv = nn.ConvTranspose2d(nf_to_hidden, nf_to_obs, kernel_size=stride_to_obs, stride=stride_to_obs)
        
    def forward(self, z, v, canvas, h, c):
        K = v.size(1)
        z = z.contiguous().view(-1, 1, z.size(1), z.size(2), z.size(3)).repeat(1, v.size(1), 1, 1, 1).view(-1, z.size(1), z.size(2), z.size(3))
        v = v.contiguous().view(-1, v.size(2), 1, 1).repeat(1, 1, z.size(2), z.size(3))
        h, c = self.lstm(torch.cat((z, v, self.conv(canvas)), dim=1), (h, c))
        canvas = canvas + self.transconv(h)
        
        return h, c, canvas
    