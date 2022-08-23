"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")


class PointNet(nn.Module):
    def __init__(self, nlatent=1024, dim_input=3, normalization='bn', activation='relu'):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """

        super(PointNet, self).__init__()
        self.dim_input = dim_input
        if normalization == 'sn':
            self.conv1 = SpectralNorm(torch.nn.Conv1d(dim_input, 64, 1))
            self.conv2 = SpectralNorm(torch.nn.Conv1d(64, 128, 1))
            self.conv3 = SpectralNorm(torch.nn.Conv1d(128, nlatent, 1))
            self.lin1 = SpectralNorm(nn.Linear(nlatent, nlatent))
            self.lin2 = SpectralNorm(nn.Linear(nlatent, nlatent))
        else:
            self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
            self.lin1 = nn.Linear(nlatent, nlatent)
            self.lin2 = nn.Linear(nlatent, nlatent)

        norm = torch.nn.BatchNorm1d if normalization == 'bn' else nn.Identity
        self.bn1 = norm(64)
        self.bn2 = norm(128)
        self.bn3 = norm(nlatent)
        self.bn4 = norm(nlatent)
        self.bn5 = norm(nlatent)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.nlatent = nlatent

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = self.activation(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = self.activation(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)

    def return_by_layer(self, x):
        out = []
        out.append(self.activation(self.bn1(self.conv1(x))))
        out.append(self.activation(self.bn2(self.conv2(out[-1]))))
        x = self.bn3(self.conv3(out[-1]))
        out.append(torch.max(x, 2)[0])
        out[-1] = out[-1].view(-1, self.nlatent)
        out.append(self.activation(self.bn4(self.lin1(out[-1]).unsqueeze(-1))))
        out.append(self.activation(self.bn5(self.lin2(out[-1].squeeze(2)).unsqueeze(-1))))
        out[-2] = out[-2].squeeze(2)
        out[-1] = out[-1].squeeze(2)
        return out


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, normalization='bn', rate=0.5, activation='relu'):
        super(LinearBlock, self).__init__()

        # initialize fully connected layer
        if normalization == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=True))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = torch.nn.Dropout(rate)

        # initialize normalization
        norm_dim = output_dim
        if normalization == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif normalization == 'none' or normalization == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(normalization)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):

        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_blocks, normalization='bn', rate=0.5, activation='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_size, hidden_size, normalization, rate, activation)]
        for i in range(num_blocks - 2):
            self.model += [LinearBlock(hidden_size, hidden_size, normalization, rate, activation)]
        self.model += [LinearBlock(hidden_size, output_size, normalization, rate=0.0, activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def get_num_ada_norm_params(model):
    # return the number of AdaNorm parameters needed by the model
    num_ada_norm_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveBatchNorm1d" or m.__class__.__name__ == "AdaptiveInstanceNorm":
            num_ada_norm_params += 2 * m.norm.num_features
    return num_ada_norm_params


class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(num_features, eps, momentum, affine)

    def forward(self, x, params):
        a = params[:, :params.size(1) // 2].unsqueeze(2)
        b = params[:, params.size(1) // 2:].unsqueeze(2)
        return a * x + b * self.norm(x)  # TODO(msegu): ouch, why a * x and not just a? Must be a bug


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        self.bottleneck_size = opt.bottleneck_size
        self.style_bottleneck_size = opt.style_bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers
        self.num_layers_style = opt.num_layers_style
        self.decode_style = opt.decode_style
        self.generator_norm = opt.generator_norm

        super(Mapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {opt.hidden_neurons}, num_layers {opt.num_layers}, "
            f"num_layers_style {opt.num_layers_style}, activation {opt.activation}")
        if self.generator_norm == 'sn':
            self.conv1 = SpectralNorm(torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1))
            self.conv2 = SpectralNorm(torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1))
            self.conv_list = nn.ModuleList(
                [SpectralNorm(torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1)) for i in range(self.num_layers)])
            self.conv_list_style = nn.ModuleList(
                [SpectralNorm(torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1)) for i in range(self.num_layers_style)])
            self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)
        else:
            self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
            self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
            self.conv_list = nn.ModuleList(
                [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])
            self.conv_list_style = nn.ModuleList(
                [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers_style)])
            self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        norm = torch.nn.BatchNorm1d if self.generator_norm == 'bn' else nn.Identity
        self.bn1 = norm(self.bottleneck_size)
        self.bn2 = norm(self.hidden_neurons)
        self.bn_list = nn.ModuleList([norm(self.hidden_neurons) for i in range(self.num_layers)])
        self.bn_list_style = nn.ModuleList(
            [norm(self.hidden_neurons) for i in range(self.num_layers_style)])

        self.activation = get_activation(opt.activation)

    def forward(self, x, content, style):
        x = self.conv1(x) + content
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))

        if self.decode_style:
            x = x + style
            for i in range(self.num_layers_style):
                x = self.activation(self.bn_list_style[i](self.conv_list_style[i](x)))
        return self.last_conv(x)


class AdaptiveMapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers
        self.num_layers_style = opt.num_layers_style
        self.decode_style = opt.decode_style

        super(AdaptiveMapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {opt.hidden_neurons}, num_layers {opt.num_layers}, "
            f"activation {opt.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = AdaptiveBatchNorm1d(self.bottleneck_size)
        self.bn2 = AdaptiveBatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([AdaptiveBatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(opt.activation)

    def forward(self, x, content, adabn_params):
        x = self.conv1(x) + content
        x = self.activation(self.bn1(x, adabn_params[:, 0:self.bottleneck_size * 2]))
        x = self.activation(self.bn2(
            self.conv2(x), adabn_params[:,
                           self.bottleneck_size * 2:
                           self.bottleneck_size * 2 + self.hidden_neurons * 2]))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](
                self.conv_list[i](x),
                adabn_params[:,
                self.bottleneck_size * 2 + (1 + i) * self.hidden_neurons * 2:
                self.bottleneck_size * 2 + (2 + i) * self.hidden_neurons * 2]))

        return self.last_conv(x)


class Mapping3Dto3D(nn.Module):
    """
    Core StyleAtlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Mattia Segu' 01.11.2019
    """

    def __init__(self, opt):
        self.opt = opt
        self.bottleneck_size = opt.style_bottleneck_size
        self.input_size = 3
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers_style
        super(Mapping3Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {opt.hidden_neurons}, num_layers {opt.num_layers}, activation {opt.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.opt.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
