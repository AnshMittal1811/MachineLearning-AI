import numpy as np
'''Pytorch implementations of hyper-network modules.'''
from torch import nn
import torch
from torchmeta.modules import (MetaModule, MetaSequential)
import custom_layers
from collections import OrderedDict


class LowRankHyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False,
                 rank=10, nonlinearity='relu'):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        self.hypo_parameters = dict(hypo_module.meta_named_parameters())

        # self.embedding_net = modules.FCBlock(in_features=hyper_in_features, out_features=hyper_hidden_features,
        #                                      num_hidden_layers=1, hidden_ch=hyper_hidden_features)
        self.representation_dim = 0

        self.rank = rank
        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in self.hypo_parameters.items():
            self.names.append(name)
            self.param_shapes.append(param.size())

            out_features = int(torch.prod(torch.tensor(param.size()))) if 'bias' in name else param.shape[0]*rank + param.shape[1]*rank
            self.representation_dim += out_features

            hn = custom_layers.FCBlock(in_features=hyper_in_features, out_features=out_features,
                                       num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                       outermost_linear=True, norm=None, nonlinearity=nonlinearity)
            if 'bias' in name:
                hn.net[-1].bias.data = torch.zeros_like(hn.net[-1].bias.data)
            else:
                hn.net[-1].bias.data = torch.ones_like(hn.net[-1].bias.data) / np.sqrt(self.rank)
            hn.net[-1].weight.data *= 1e-1

            self.nets.append(hn)

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        # embedding = self.embedding_net(z)

        params = OrderedDict()
        representation = []
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            low_rank_params = net(z)
            representation.append(low_rank_params)

            if 'bias' in name:
                batch_param_shape = (-1,) + param_shape
                params[name] = self.hypo_parameters[name] + low_rank_params.reshape(batch_param_shape)
            else:
                a = low_rank_params[:, :self.rank*param_shape[0]].view(-1, param_shape[0], self.rank)
                b = low_rank_params[:, self.rank*param_shape[0]:].view(-1, self.rank, param_shape[1])
                low_rank_w = a.matmul(b)
                params[name] = self.hypo_parameters[name] * low_rank_w
                # params[name] = self.hypo_parameters[name] * torch.sigmoid(low_rank_w)

        return params


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module,siren=False):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = custom_layers.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                       num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                       outermost_linear=True, norm='layernorm')
            if 'weight' in name:
                hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1], siren=siren))
            elif 'bias' in name:
                hn.net[-1].apply(lambda m: hyper_bias_init(m, siren=siren))

            self.nets.append(hn)


    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class FILMNetwork(nn.Module):
    def __init__(self, hypo_module, latent_dim, num_hidden=3):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = custom_layers.FCBlock(in_features=latent_dim, out_features=int(2*torch.tensor(param.shape[0])),
                                       num_hidden_layers=num_hidden, hidden_ch=latent_dim, outermost_linear=True,
                                       nonlinearity='relu')
            # hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            self.nets.append(hn)

    def forward(self, z):
        params = []
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            net_out = net(z)
            layer_params = {}
            layer_params['gamma'] = net_out[:, :param_shape[0]].unsqueeze(1) + 1
            layer_params['beta'] = net_out[:, param_shape[0]:].unsqueeze(1)
            params.append(layer_params)

        return params


############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1e1

    if hasattr(m, 'bias') and siren:
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e1

    if hasattr(m, 'bias') and siren:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
