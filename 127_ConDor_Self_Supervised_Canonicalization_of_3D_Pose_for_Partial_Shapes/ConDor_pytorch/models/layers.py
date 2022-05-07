import torch.nn as nn
import torch
from spherical_harmonics.spherical_cnn import SphericalHarmonicsCoeffs


def apply_layers(x, layers):
    y = dict()
    # print(x.keys())
    for l in x:
        # print(x[l].shape, l)
        if l.isnumeric():
            y[l] = layers[int(l)](x[l])
    return y

def type_0(x, S2):
    """
    Spherical Harmonics Transform to extract type 0 features
    """

    y = SphericalHarmonicsCoeffs(l_list=[0], base=S2).compute(x)
    return y['0']



def type_1(x, S2):
    """
    Spherical Harmonics Transform to extract type 1 features that are equivariant to SO(3)
    """
    y = SphericalHarmonicsCoeffs(l_list=[1], base=S2).compute(x)
    return y['1']


def set_sphere_weights(in_channel, out_channel, types):
    weights = []
    for l in types:
        if int(l) == 0:
            weights.append(nn.Linear(in_channel[int(l)], out_channel))
        else:
            weights.append(nn.Linear(in_channel[int(l)], out_channel, bias=False))
    return torch.nn.Sequential(*weights)

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, bn_momentum = 0.75, apply_norm = True, activation = None):
        super(MLP, self).__init__()
        
        self.mlp = nn.Linear(in_channels, out_channels)
        if apply_norm:
            self.batchnorm = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation
        self.apply_norm = apply_norm
        
    def forward(self, x):
        out = self.mlp(x)        
        if self.activation is not None:
            out = self.activation(out)
        if self.apply_norm:
            out = self.batchnorm(out.transpose(1, -1)).transpose(1, -1)

        return out

class MLP_layer(nn.Module):
    def __init__(self, in_channels, units = [32, 64, 128], bn_momentum = 0.75, apply_norm = False, activation = None):
        super(MLP_layer, self).__init__()
        
        self.input_layer = nn.Linear(in_channels, units[0])
        self.mlp = []
        self.batchnorm = []
        for i in range(1, len(units)):
            self.mlp.append(MLP(units[i-1], units[i], bn_momentum = bn_momentum, apply_norm = apply_norm, activation = activation))

        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        
        out = self.input_layer(x)
        out = self.mlp(out)
        return out