import torch
import math
import numpy as np
from torch import nn
from torchmeta.modules import MetaModule
from collections import OrderedDict
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diff_operators import jacobian
from autoint import autograd_modules
from autoint.session import Session


def init_weights_requ(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1/math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5*self.relu(input)**2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input)-self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


def layer_factory(layer_type):
    layer_dict = \
        {
         'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'sigmoid': (nn.Sigmoid(), None),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
        }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu',
                 weight_init=None, w0=30, set_bias=None,
                 dropout=0.0):
        super().__init__()

        self.first_layer_init = None
        self.dropout = dropout

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []
            for i in range(num_hidden_layers+1):
                hidden_features.append(num_hidden_features)
        else:
            num_hidden_layers = len(hidden_features)-1
        print(f"net_size={hidden_features}")

        # Create the net
        print(f"num_layers={len(hidden_features)}")
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to " \
                                                              "match the length of the list of non-linearities"

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                layer_factory(nonlinearity[0])[0]
            ))
            for i in range(num_hidden_layers):
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    layer_factory(nonlinearity[i+1])[0]
                ))

            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))
        elif isinstance(nonlinearity, str):
            nl, weight_init = layer_factory(nonlinearity)
            if(nonlinearity == 'sine'):
                first_nl = FirstSine()
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                first_nl
            ))

            for i in range(num_hidden_layers):
                if(self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    copy.deepcopy(nl)
                ))

            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))
            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    copy.deepcopy(nl)
                ))

        self.net = nn.Sequential(*self.net)

        if isinstance(nonlinearity, list):
            for layer_num, layer_name in enumerate(nonlinearity):
                self.net[layer_num].apply(layer_factory(layer_name)[1])
        elif isinstance(nonlinearity, str):
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if self.first_layer_init is not None:
                self.net[0].apply(self.first_layer_init)

        if set_bias is not None:
            self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords):
        output = self.net(coords)
        return output


class CoordinateNet(nn.Module):
    '''A canonical coordinate network'''
    def __init__(self, out_features=1, nl='sine', in_features=3,
                 hidden_features=256, num_hidden_layers=3, num_pe_fns=6,
                 use_grad=True, w0=30, grad_var=None, input_processing_fn=None):
        super().__init__()
        self.use_grad = use_grad
        self.grad_var = grad_var
        self.input_processing_fn = input_processing_fn

        if use_grad:
            normalize_pe = True
            assert grad_var is not None
        else:
            normalize_pe = False

        self.nl = nl
        if self.nl != 'sine':
            in_features = in_features * (2*num_pe_fns + 1)

        self.pe = PositionalEncoding(num_encoding_functions=num_pe_fns, normalize=normalize_pe)
        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=nl,
                           w0=w0)
        print(self)

    def forward(self, model_input):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}

        if self.input_processing_fn is not None:
            input_dict_transformed = self.input_processing_fn(input_dict)
        coords = input_dict_transformed['coords']

        if self.nl != 'sine':
            coords_pe = self.pe(coords)
            output = self.net(coords_pe)
        else:
            output = self.net(coords)

        if self.use_grad:
            output = jacobian(output, input_dict_transformed[self.grad_var])[0][:, :, 0]

        return {'model_in': input_dict_transformed, 'model_out': {'output': output}}


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class RadianceNet(MetaModule):
    def __init__(self, out_features=1, hidden_layers=6, hidden_features=256,
                 input_name=['ray_samples', 'ray_orientations'],
                 input_pe_params={'ray_samples': 10, 'ray_orientations': 4},
                 nl='swish',
                 input_processing_fn=None,
                 sampler=None,
                 normalize_pe=True,
                 use_grad=True):

        super().__init__()
        self.input_name = input_name
        self.input_dim = {'t': 1,
                          'ray_directions': 3,
                          'ray_origins': 3,
                          'ray_orientations': 6,
                          'ray_samples': 3}
        self.input_pe_params = input_pe_params
        self.input_processing_fn = input_processing_fn
        self.sampler = sampler
        self.normalize_pe = normalize_pe
        self.use_grad = use_grad
        self.session = Session()

        # params
        self.in_features = 0
        for key in input_name:
            self.in_features += self.input_dim[key] * (1 + 2 * self.input_pe_params[key])

        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.nl = nl

        # run forward pass to bootstrap session
        self.trace_graph()
        self.session = self.session.cuda()
        self.backward_session = self.session.get_backward_graph().cuda()
        self.backward_session.preprocess()

        if self.use_grad:
            self.set_mode('grad')
        else:
            self.set_mode('integral')

    def get_nl_fn(self, name):
        d = {'swish': autograd_modules.Swish,
             'sine': autograd_modules.Sine,
             'requ': autograd_modules.ReQU}
        return d[name]

    def trace_graph(self):
        x1 = autograd_modules.Value(torch.ones(1, 1), self.session)
        x2 = autograd_modules.Value(torch.ones(1, 1), self.session)
        x3 = autograd_modules.Value(torch.ones(1, 1), self.session)
        x4 = autograd_modules.Value(torch.ones(1, 1), self.session)

        # prep inputs
        # t, ray_dirs, origins, orientations
        t = autograd_modules.Input(torch.Tensor(1, 1, self.input_dim['t']), id='t')(x1)
        ray_dirs = autograd_modules.Constant(torch.Tensor(1, 1, self.input_dim['ray_directions']), id='ray_directions')(x2)

        # calculate ray_samples  as function of t, ray_dirs, and origins
        scaled_dir = autograd_modules.HadamardProd()(ray_dirs, t)
        ray_origins = autograd_modules.Constant(torch.Tensor(1, 1, self.input_dim['ray_origins']), id='ray_origins')(x3)
        ray_samples = autograd_modules.HadamardAdd()(ray_origins, scaled_dir)

        # calculate positional encodings
        ray_samples_pe = autograd_modules.PositionalEncoding(normalize=self.normalize_pe,
                                                             num_encoding_functions=self.input_pe_params['ray_samples'])(ray_samples)

        if 'ray_orientations' in self.input_name:
            orientations = autograd_modules.Constant(torch.Tensor(1, 1, self.input_dim['ray_orientations']), id='ray_orientations')(x4)
            orientations_pe = autograd_modules.PositionalEncoding(normalize=self.normalize_pe,
                                                                  num_encoding_functions=self.input_pe_params['ray_orientations'])(orientations)

            # send through hidden layers
            out = autograd_modules.Concatenate(num_inputs=2)(ray_samples_pe, orientations_pe)
        else:
            out = ray_samples_pe

        if not isinstance(self.nl, list):
            net = []
            net.append(autograd_modules.Linear(self.in_features, self.hidden_features, nl=self.nl))

            nl_fn = self.get_nl_fn(self.nl)
            net.append(nl_fn())
            for i in range(self.hidden_layers):
                net.append(autograd_modules.Linear(self.hidden_features, self.hidden_features, nl=self.nl))
                net.append(nl_fn())
            net.append(autograd_modules.Linear(self.hidden_features, self.out_features, nl=self.nl))
            net = nn.Sequential(*net)
        else:
            if len(self.hidden_features) != len(self.nl):
                print("Provided a different number of nl and feature sizes")

            net = []

            net.append(autograd_modules.Linear(self.in_features, self.hidden_features[0], nl=self.nl[0]))
            nl_fn = self.get_nl_fn(self.nl[0])
            net.append(nl_fn())
            print(f"Linear {self.in_features}x{self.hidden_features[0]}")
            print(f"{self.nl[0]}")

            for i in range(1, len(self.hidden_features)):
                net.append(autograd_modules.Linear(self.hidden_features[i-1], self.hidden_features[i], nl=self.nl[i]))
                nl_fn = self.get_nl_fn(self.nl[i])
                net.append(nl_fn())
                print(f"Linear {self.hidden_features[i - 1]}x{self.hidden_features[i]}")
                print(f"NL {self.nl[i]}")

            net.append(autograd_modules.Linear(self.hidden_features[-1], self.out_features, nl=self.nl[-1]))
            print(f"Linear {self.hidden_features[-1]}x{self.out_features}")
            net = nn.Sequential(*net)

        net(out)

    def set_mode(self, mode):
        if mode not in ['grad', 'integral']:
            raise ValueError("argument must be 'grad' or 'integral'")
        self.mode = mode

    def forward(self, model_input, params=None):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}

        input_dict['params'] = params

        if self.input_processing_fn is None:
            input_dict_transformed = input_dict
        else:
            input_dict_transformed = self.input_processing_fn(input_dict, sampler=self.sampler,
                                                              return_posts=self.mode == 'integral' and self.use_grad)

        if self.mode == 'grad':
            out = self.backward_session.compute_graph_fast(input_dict_transformed)
        elif self.mode == 'integral':
            out = self.session.compute_graph_fast(input_dict_transformed)

        output_dict = {'output': out}
        return {'model_in': input_dict_transformed, 'model_out': output_dict}


def input_processing_fn(input_dict, sampler=None, sampling_interval=(2, 6), return_posts=False):
    t = input_dict['t']
    t_intervals = t[..., 1:, :] - t[..., :-1, :]

    t_intervals = torch.cat((t_intervals, 1e10*torch.ones_like(t_intervals[:, 0:1, :])), dim=-2)
    input_dict['t_intervals'] = t_intervals

    if sampler is not None:
        input_dict = sampler(input_dict, return_posts=return_posts)

    t_intervals = input_dict['t_intervals']
    t = input_dict['t']

    t_intervals = t_intervals * input_dict['ray_directions'].norm(p=2, dim=-1)[..., None]

    num_samples = input_dict['t'].shape[-2]
    origins = input_dict['ray_origins'].repeat(1, num_samples, 1)
    directions = input_dict['ray_directions'].repeat(1, num_samples, 1)
    orientations = input_dict['ray_orientations'].repeat(1, num_samples, 1).detach().requires_grad_(True)

    ray_samples = origins + t * directions
    out = input_dict
    out.update({'ray_samples': ray_samples, 'ray_orientations': orientations,
                't': t, 't_intervals': t_intervals, 'ray_origins': origins})

    return out


class SamplingNet(nn.Module):
    def __init__(self, Nt=128, ncuts=32, num_hidden_layers=4,
                 hidden_features=256, set_bias=0.0, w0=30,
                 nonlinearity='relu',
                 sampling_interval=(2., 6.)):
        super().__init__()
        self.range = sampling_interval[1] - sampling_interval[0]
        self.sampling_interval = sampling_interval
        self.ncuts = ncuts
        self.default_interval = (sampling_interval[1] - sampling_interval[0]) / self.ncuts
        self.Nt = Nt

        num_inputs = 6
        num_outputs = self.ncuts

        self.use_pe = True
        if self.use_pe:
            num_encoding_fns = 5
            self.positional_encoding_fn = PositionalEncoding(num_encoding_functions=num_encoding_fns,
                                                             input_dim=3,
                                                             normalize=False,
                                                             gaussian_pe=False)
            num_inputs = num_inputs*(1+2*num_encoding_fns)

        self.net = FCBlock(in_features=num_inputs, out_features=num_outputs,
                           num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                           outermost_linear=True, nonlinearity=nonlinearity,
                           w0=w0, set_bias=set_bias)

    def sample_stratified_between_posts(self, input_posts, num_samples_per_interval, near=2., far=6.):
        posts = input_posts['t']  # N*R,Posts,1
        posts_shape_o = list(posts.shape)
        posts_shape_o[-2] = 1
        interval_lows = torch.cat((near*torch.ones(posts_shape_o).to(posts.device),
                                  posts), dim=-2)
        interval_lows = interval_lows.unsqueeze(-2)  # N*R,Intervals,1 -> N*R,Intervals,SamplesPerInterval,1

        interval_highs = torch.cat((posts,
                                    far*torch.ones(posts_shape_o).to(posts.device)), dim=-2)
        interval_highs = interval_highs.unsqueeze(-2)  # N*R,Intervals,1 -> N*R,Intervals,SamplesPerInterval,1

        interval_length = interval_highs-interval_lows

        posts_shape_n = list(posts.shape)
        posts_shape_n[-2] = num_samples_per_interval
        t = interval_lows + interval_length*torch.arange(0., 1., 1./num_samples_per_interval).reshape(1, 1, -1, 1).to(posts.device)
        t += torch.rand_like(t).to(posts.device)*interval_length/num_samples_per_interval

        input_posts.update({'t': t.reshape(t.shape[0], -1, 1)})  # N*R, Intervals, SamplesPerIntervals, 1 -> N*R, Samples, 1
        return input_posts

    def forward(self, model_input, return_posts=False):
        if self.use_pe:
            model_in = torch.cat([self.positional_encoding_fn(model_input['ray_directions']),
                                  self.positional_encoding_fn(model_input['ray_origins'])],
                                 dim=-1)
        else:
            model_in = torch.cat((model_input['ray_directions'], model_input['ray_origins']), dim=-1)

        intervals = torch.abs(self.net(model_in) + self.default_interval).permute(0, 2, 1)

        # normalize intervals to correct range
        intervals = intervals * self.range / torch.sum(intervals, dim=-2, keepdim=True)

        # compute post locations
        posts = torch.cat((self.sampling_interval[0]*torch.ones_like(intervals[..., :1, :]),
                           self.sampling_interval[0] + torch.cumsum(intervals, dim=-2)), dim=-2)

        # stratified sampling in between posts
        samples_per_interval = self.Nt // self.ncuts
        if return_posts:
            t = posts
        else:
            t = self.sample_stratified_between_posts({'t': posts[..., 1:-1, :]}, samples_per_interval,
                                                     near=self.sampling_interval[0],
                                                     far=self.sampling_interval[1])['t']

        model_input['t'] = t
        t_intervals = t[..., 1:, :] - t[..., :-1, :]
        t_intervals = torch.cat((t_intervals, 1e10*torch.ones_like(t_intervals[:, 0:1, :])), dim=-2)
        model_input['t_intervals'] = t_intervals
        return model_input
