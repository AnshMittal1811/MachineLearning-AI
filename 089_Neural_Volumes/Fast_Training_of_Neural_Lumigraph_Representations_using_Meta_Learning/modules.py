import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import utils.activations as activations
import utils.common_utils as common_utils
import utils.diff_operators as diff_operators
import utils.math_utils_torch as mut


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if 'weight_g' in params:
            # Denormalize weight_norm.
            from torch import _weight_norm
            hook = next(hook for hook in self._forward_pre_hooks.values() if hook.name == 'weight')
            weight = _weight_norm(params['weight_v'], params['weight_g'], hook.dim)
        else:
            weight = params['weight']

        bias = params.get('bias', None)

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class BatchConv2d(nn.Conv2d, MetaModule):
    '''A convolutional meta-layer for dealing with batched weight matrices and biases'''
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params['weight']
        bias = params.get('bias', None)

        output = F.conv2d(input, weight, bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class SkipConnection(MetaModule):
    """
    Bypasses features.
    """

    def __init__(self, inner: MetaModule):
        super().__init__()
        self.inner = inner

    def forward(self, x, params=None):
        if isinstance(self.inner, MetaModule):
            y = self.inner(x, params=get_subdict(params, 'inner'))
        elif isinstance(self.inner, nn.Module):
            y = self.inner(x)
        else:
            raise TypeError('The module must be either a torch module '
                            '(inheriting from `nn.Module`), or a `MetaModule`. '
                            'Got type: `{0}`'.format(type(self.inner)))

        y = torch.cat((x, y), -1)
        return y


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, activation='relu', activation_last='none', skip_connections=[]):
        super().__init__()

        self.first_layer_init = None

        _, self.weight_init, self.first_layer_init = activations.nls_and_inits[activation]
        act_last = None
        if activation_last != 'none':
            act_last = activations.get_activation(activation_last, n_dims=hidden_features)
        elif not outermost_linear:
            act_last = activations.get_activation(activation, n_dims=hidden_features)

        # We currently only support a single skip connection.
        if len(skip_connections) == 0:
            skip_connections = [-1, -1]
        else:
            skip_connections = skip_connections[0]

        # layers 0 ... last
        total_layers = 1 + num_hidden_layers + 1
        self.net = []
        self.skips = []
        skip_in = in_features
        # Combine all the layers into a nested
        # sequential network.
        for i in range(total_layers):
            # In/out
            layer_in = hidden_features
            if i == 0:
                layer_in = in_features
            layer_out = hidden_features
            if i == total_layers - 1:
                layer_out = out_features

            # Before skip connection.
            if i == skip_connections[0]:
                skip_in = layer_in

            # Output of the skipped block.
            if i == skip_connections[1] - 1:
                layer_out -= skip_in

            # Activation
            modules = [BatchLinear(layer_in, layer_out)]
            if i == total_layers - 1:
                # Last layer
                if act_last is not None:
                    modules += [act_last]
            else:
                modules += [activations.get_activation(activation, n_dims=hidden_features)]
            # Executes modules sequentially.
            layer = MetaSequential(*modules)

            if i >= skip_connections[0] and i < skip_connections[1]:
                # Skip connection body.
                self.skips += [layer]
                if i == skip_connections[1] - 1:
                    # Skip finished.
                    self.net += [SkipConnection(MetaSequential(*self.skips))]
            else:
                # Normal layer.
                self.net += [layer]
        self.net = MetaSequential(*self.net)

        # Reset weights.
        self.reinitialize()

    def reinitialize(self, weight_init=None, first_layer_init=None, last_layer_init=None):
        """
        Initializes the weights.
        """
        # Defaults.
        weight_init = self.weight_init if weight_init is None else weight_init
        first_layer_init = self.first_layer_init if first_layer_init is None else first_layer_init

        # Resolve missing.
        assert weight_init is not None
        if first_layer_init is None:
            first_layer_init = weight_init
        if last_layer_init is None:
            last_layer_init = weight_init

        lin_modules = [m for m in self.net.modules() if type(m) in [nn.Linear, BatchLinear]]

        # Apply special initialization to first layer, if applicable.
        [first_layer_init(m) for m in lin_modules[:1]]
        # Middle layers.
        [weight_init(m) for m in lin_modules[1:-1]]
        # Apply special initialization to last layer, if applicable.
        [last_layer_init(m) for m in lin_modules[-1:]]

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        acts = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        acts['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                acts['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return acts


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, opt,
                 in_features=2, hidden_features=256, out_features=1,
                 num_hidden_layers=3, skip_connections=[],
                 activation='sine', activation_last='none',
                 positional_encoding=None, positional_encoding_kwargs={}):
        super().__init__()
        self.opt = opt
        self.activation = activation
        self.activation_last = activation_last
        self.in_features = in_features
        self.out_features = out_features

        # Use pos encoding for NERF network or on explicit demand for any network.
        self.positional_encoding = None
        if positional_encoding is not None and positional_encoding != 'none':
            self.positional_encoding = self.get_positional_encoding(
                positional_encoding, in_features, positional_encoding_kwargs)
            in_features = self.positional_encoding.out_dim

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True,
                           activation=activation, activation_last=activation_last,
                           skip_connections=skip_connections)

        self.flow_decoder = []

    @ classmethod
    def get_positional_encoding(cls, positional_encoding: str, in_features: int, enc_kwargs: dict) -> nn.Module:
        """
        Gets proper pos enc layer.
        """
        modules = {
            'nerf': PosEncodingNeRF,
            'idr': PosEncodingIDR,
            'ff': PosEncodingFF,
        }

        enc_kwargs['in_features'] = in_features
        return modules[positional_encoding](**enc_kwargs)

    def _apply_flow(self, coords: torch.Tensor, times: torch.Tensor = None) -> torch.Tensor:
        """
        Only applies flow if time is provided.
        """
        if not self.flow_decoder:
            # Without flow decoder, do nothing.
            return coords

        if coords.shape[-1] != 4 and times is None:
            # Without time do nothing.
            return coords

        res = self.flow_decoder[0]({
            'coords': coords,
            'time': times,
        })
        return self.flow_decoder[0].apply_flow_x(coords, times)

    def forward(self, model_input, params=None):
        raise RuntimeError('Use specialized class.')

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

    @ property
    def device(self):
        """
        CUDA or CPU?
        """
        return next(self.parameters()).device


class SDFDecoder(SingleBVPNet):
    """
    Specialization for the shape.
    """

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Input coords.
        model_in = model_input['coords']
        coords = model_in

        # Apply flow if possible.
        coords = self._apply_flow(coords, model_input.get('time', None))

        # Optional positional encoding.
        if self.positional_encoding is not None:
            coords = self.positional_encoding(coords)

        # The core net.
        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': model_in, 'model_out': output}


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=False,
                 num_bands: int = 0, **kwargs):
        super().__init__()

        self.in_features = in_features

        if num_bands > 0:
            # User specified.
            self.num_frequencies = num_bands
        elif self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

    def extra_repr(self) -> str:
        return f'freq={self.num_frequencies}, in_features={self.in_features}, out_features={self.out_dim}'


class PosEncodingIDR(nn.Module):
    '''Module to add positional encoding as in IDR [Lipman et al. 2020].'''

    def __init__(self, in_features: int, num_bands: int, channels=None, bands_exp=False, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.num_bands = num_bands
        self.bands_exp = bands_exp

        if channels is None:
            self.channels = np.arange(in_features)
        else:
            self.channels = np.array(channels, int)
        self.out_dim = in_features + 2 * len(self.channels) * num_bands

    def forward(self, coords):
        # Only encode selected.
        coords_select = torch.stack([coords[..., i] for i in self.channels], -1)

        res = [coords]
        for i in range(self.num_bands):
            # Note, the paper says 2*i (not 2**i) but I guess that is a typo as confirmed by Lior.
            if self.bands_exp:
                arg = 2 ** i * np.pi * coords_select
            else:
                arg = 2 * (i + 1) * np.pi * coords_select
            res += [torch.sin(arg)]
            res += [torch.cos(arg)]

        return torch.cat(res, -1)

    def extra_repr(self) -> str:
        mode = 'exp' if self.bands_exp else 'lin'
        return f'bands={self.num_bands} ({mode}), in_features={self.in_features}, out_features={self.out_dim}'


class PosEncodingFF(nn.Module):
    '''Module to add Fourier Features as in [Tancik et al. 2020].'''

    def __init__(self, in_features: int, num_bands: int, channels=None, sigma=1.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.num_bands = num_bands

        if channels is None:
            self.channels = np.arange(in_features)
        else:
            self.channels = np.array(channels, int)

        self.sigma = sigma
        if np.isscalar(sigma):
            sigma = [sigma] * len(self.channels)
        else:
            assert len(sigma) == len(self.channels)

        B = np.random.normal(scale=sigma, size=[num_bands, len(self.channels)]).T
        self.register_buffer('B', torch.from_numpy(B).float())

        self.out_dim = in_features + 2 * num_bands

    def forward(self, coords):
        # Only encode selected.
        coords_select = torch.stack([coords[..., i] for i in self.channels], -1)
        XB = torch.matmul(coords_select, self.B)
        coords_encoded = (torch.sin(XB), torch.cos(XB))

        # Concatenate all orign and sselected encoded features
        return torch.cat([coords, *coords_encoded], dim=-1)

    def extra_repr(self) -> str:
        return f'bands={self.num_bands}, in_features={self.in_features}, sigma={self.sigma}, out_features={self.out_dim}'

######################
# Tools
######################


def batch_decode(decoder, model_input, batch_size: int, params=None,
                 callback=None, callback_args=None, verbose=False,
                 out_feature_slice: slice = None, return_inputs=True):
    """
    Executes network in small batches.
    Only makes sense without gradients.
    """
    coords = model_input['coords']
    input_device = coords.device

    if callback is None and (batch_size <= 0 or batch_size >= coords.shape[1]):
        # Shortcut.
        output = decoder.forward(model_input, params=params)
        if out_feature_slice is not None:
            output['model_out'] = output['model_out'][..., out_feature_slice]
        return output

    outputs = []
    n_batches = int(np.maximum(np.ceil(coords.shape[1] / batch_size), 1))
    for i in range(n_batches):

        # Slice input.
        x = {k: v for k, v in model_input.items()}
        x_start = (i * batch_size)
        x_end = ((i + 1) * batch_size)
        x['coords'] = x['coords'][:, x_start:x_end, ...].to(decoder.device)
        if 'time' in x:
            x['time'] = x['time'][:, x_start:x_end, ...].to(decoder.device)
        if verbose:
            print(f'\tDecoding batch {i} / {n_batches} (samples {x_start} -- x_end)')

        # Execute batch.
        output = decoder.forward(x, params=params)
        # Callback?
        if callback is not None:
            output = callback(i, x, output, callback_args)
            if output is None:
                # We are to ignore output.
                continue

        # Save "some" memory.
        if not return_inputs:
            del output['model_in']

        # Filter output.
        if out_feature_slice is not None:
            output['model_out'] = output['model_out'][..., out_feature_slice]

        # Back to input device.
        for k in output.keys():
            output[k] = output[k].to(input_device)

        # Collect outputs.
        outputs += [output]
        if len(outputs) >= 100:
            # Concatenate outputs.
            res = {}
            for k in outputs[0].keys():
                res[k] = torch.cat([output[k] for output in outputs], 1)
            outputs = [res]

    # No outputs.
    if len(outputs) == 0:
        return None

    # Concatenate outputs.
    res = {}
    for k in outputs[0].keys():
        res[k] = torch.cat([output[k] for output in outputs], 1)

    # Keep the original input as input.
    # Important for differentiability.
    res['model_in'] = coords

    return res
