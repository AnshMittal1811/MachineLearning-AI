import torch
from torch import nn
from torch.distributions import Normal

from conv_lstm import Conv2dLSTMCell

class Prior(nn.Module):
    def __init__(self, stride_to_hidden, nf_to_hidden, nf_enc, nf_z):
        super(Prior, self).__init__()
        self.conv1 = nn.Conv2d(32, nf_enc, kernel_size=stride_to_hidden, stride=stride_to_hidden)
        self.lstm = Conv2dLSTMCell(nf_enc+nf_z, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(nf_to_hidden, 2*nf_z, kernel_size=5, stride=1, padding=2)
        
    def forward(self, r, z, h, c):
        lstm_input = self.conv1(r)
        h, c = self.lstm(torch.cat((lstm_input, z), dim=1), (h, c))
        mu, logvar = torch.split(self.conv2(h), z.size(1), dim=1)
        std = torch.exp(0.5*logvar)
        p = Normal(mu, std)
        
        return h, c, p

class Posterior(nn.Module):
    def __init__(self, stride_to_hidden, nf_to_hidden, nf_enc, nf_z):
        super(Posterior, self).__init__()
        self.conv1 = nn.Conv2d(2*32, nf_enc, kernel_size=stride_to_hidden, stride=stride_to_hidden)
        self.lstm = Conv2dLSTMCell(nf_enc+nf_z, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(nf_to_hidden, 2*nf_z, kernel_size=5, stride=1, padding=2)
        
    def forward(self, r, r_prime, z, h, c):
        lstm_input = self.conv1(torch.cat((r, r_prime), dim=1))
        h, c = self.lstm(torch.cat((lstm_input, z), dim=1), (h, c))
        mu, logvar = torch.split(self.conv2(h), z.size(1), dim=1)
        std = torch.exp(0.5*logvar)
        p = Normal(mu, std)
        
        return h, c, p
    