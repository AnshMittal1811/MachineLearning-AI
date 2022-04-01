import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms.functional as TVF

from itertools import chain
from typing import Optional, Union
import math

from .utils import ( fourier, create_fourier_basis, smooth_min )

class PositionalEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    max_freq: float = 6.,
    N: int = 64,
    log_sampling: bool = False,
  ):
    super().__init__()
    if log_sampling:
      bands = 2**torch.linspace(1, max_freq, steps=N, requires_grad=False, dtype=torch.float)
    else:
      bands = torch.linspace(1, 2**max_freq, steps=N, requires_grad=False, dtype=torch.float)
    self.bands = nn.Parameter(bands, requires_grad=False)
    self.input_dims = input_dims
  def output_dims(self): return self.input_dims * 2 * len(self.bands)
  def forward(self, x):
    assert(x.shape[-1] == self.input_dims)
    raw_freqs = torch.tensordot(x, self.bands, dims=0)
    raw_freqs = raw_freqs.reshape(x.shape[:-1] + (-1,))
    return torch.cat([ raw_freqs.sin(), raw_freqs.cos() ], dim=-1)

class FourierEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    # TODO rename this num freqs to be more accurate.
    freqs: int = 128,
    sigma: int = 1 << 5,
    device="cpu",
  ):
    super().__init__()
    self.input_dims = input_dims
    self.freqs = freqs
    self.basis = create_fourier_basis(freqs, features=input_dims, freq=sigma, device=device)
    self.basis = nn.Parameter(self.basis, requires_grad=False)
    self.extra_scale = 1
  def output_dims(self): return self.freqs * 2
  def forward(self, x): return fourier(x, self.extra_scale * self.basis)
  def scale_freqs(self, amt: 1+1e-5, cap=2):
    self.extra_scale *= amt
    self.extra_scale = min(self.extra_scale, cap)

class LearnedFourierEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    num_freqs: int = 16,
    sigma: int = 1 << 5,
    device="cpu",
  ):
    super().__init__()
    self.input_dims = input_dims
    self.n_freqs = num_freqs
    self.basis = create_fourier_basis(num_freqs, features=input_dims, freq=sigma, device=device)
    self.basis = nn.Parameter(self.basis, requires_grad=False)
    self.extra_scale = nn.Parameter(torch.tensor(1, requires_grad=True), requires_grad=True)
  def output_dims(self): return self.n_freqs * 2
  def forward(self, x): return fourier(x, self.extra_scale * self.basis)

# It seems a cheap approximation to SIREN works just as well? Not entirely sure.
class NNEncoder(nn.Module):
  def __init__(
    self,
    input_dims: int = 3,
    out: int = 32,
    device=None,
  ):
    super().__init__()
    self.fwd = nn.Linear(input_dims, out)
  def output_dims(self): return self.fwd.out_features
  def forward(self, x):
    assert(x.shape[-1] == self.fwd.in_features)
    return torch.sin(30 * self.fwd(x))

# how to initialize the MLP, otherwise defaults to torch
mlp_init_kinds = {
  None,
  "zero",
  "kaiming",
  "siren",
  "xavier",
}

class SkipConnMLP(nn.Module):
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,

    num_layers = 5,
    hidden_size = 256,
    in_size=3, out=3,

    skip=3,
    activation = nn.LeakyReLU(inplace=True),
    latent_size=0,

    enc: Optional[Union[FourierEncoder, PositionalEncoder, NNEncoder]] = None,

    # Record the last layers activation
    last_layer_act = False,

    linear=nn.Linear,

    init=None,
  ):
    assert(init in mlp_init_kinds), "Must use init kind"
    super(SkipConnMLP, self).__init__()
    self.in_size = in_size
    map_size = 0

    self.enc = enc
    if enc is not None: map_size += enc.output_dims()

    self.dim_p = in_size + map_size + latent_size
    self.skip = skip
    self.latent_size = latent_size
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size, hidden_size,
      ) for i in range(num_layers)
    ]


    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [
      self.init.weight, self.out.weight,
      *[l.weight for l in self.layers],
    ]
    biases = [
      self.init.bias, self.out.bias,
      *[l.bias for l in self.layers],
    ]
    if init is None:
      ...
    elif init == "zero":
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "xavier":
      for t in weights: nn.init.xavier_uniform_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "siren":
      for t in weights:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(t)
        a = math.sqrt(6 / fan_in)
        nn.init._no_grad_uniform_(t, -a, a)
      for t in biases: nn.init.zeros_(t)
    elif init == "kaiming":
      for t in weights: nn.init.kaiming_normal_(t, mode='fan_out')
      for t in biases: nn.init.zeros_(t)

    self.activation = activation
    self.last_layer_act = last_layer_act

  def forward(self, p, latent: Optional[torch.Tensor]=None):
    batches = p.shape[:-1]
    init = p.reshape(-1, p.shape[-1])

    if self.enc is not None: init = torch.cat([init, self.enc(init)], dim=-1)
    if self.latent_size != 0:
      assert(latent is not None), "Did not pass latent vector when some was expected"
      init = torch.cat([init, latent.reshape(-1, self.latent_size)], dim=-1)
    else: assert((latent is None) or (latent.shape[-1] == 0)), "Passed latent vector when none was expected"

    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    if self.last_layer_act: setattr(self, "last_layer_out", x.reshape(batches + (-1,)))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))
  # smoothness of this sample along a given dimension for the last axis of a tensor
  def l2_smoothness(self, sample, values=None, noise=1e-1, dim=-1):
    if values is None: values = self(sample)
    adjusted = sample + noise * torch.rand_like(sample)
    adjusted = self(adjusted)
    return (values-adjusted).square().mean()
  def zero_last_layer(self):
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)
  def uniform_last_layer(self, a=1e-4):
    nn.init.uniform_(self.out.weight, -a, a)
    nn.init.uniform_(self.out.bias, -a, a)
  # add an additional method for capt
  def variance(self, shape=None):
    return torch.stack([l.var(shape) for l in self.layers], dim=0)

class RecurrentUnit(nn.Module):
  def __init__(
    self,
    in_size=3,
    state_size:int = 128,
    enc: Optional[Union[FourierEncoder, PositionalEncoder, NNEncoder]] = None,
  ):
    super().__init__()
    self.enc = enc
    total_in = enc.output_dims() + in_size
    self.r_i = nn.Linear(total_in, state_size)
    self.r_s = nn.Linear(state_size, state_size)

    self.z_i = nn.Linear(total_in, state_size)
    self.z_s = nn.Linear(state_size, state_size)

    self.n_i = nn.Linear(total_in, state_size)
    self.n_s = nn.Linear(state_size, state_size)
  def forward(self, x, state):
    x = torch.cat([x, self.enc(x)], dim=-1)

    reset = torch.sigmoid(self.r_i(x) + self.r_s(state))
    update = torch.sigmoid(self.z_i(x) + self.z_s(state))
    new = torch.tanh(self.n_i(x) + reset * self.n_s(state))
    new_state = (1-update) * new + update * state
    return new_state


class EncodedGRU(nn.Module):
  def __init__(
    self,
    encs: [Union[FourierEncoder, PositionalEncoder, NNEncoder]],

    state_size:int=128,

    in_size:int=3,
    out:int=3,
    latent_out:int=128,
  ):
    super(EncodedGRU, self).__init__()
    self.in_size = in_size

    self.state_size = state_size

    hidden_layers = [
      RecurrentUnit(in_size=3, state_size=state_size, enc=enc) for enc in encs
    ]
    self.layers = nn.ModuleList(hidden_layers)

    self.last = nn.Linear(state_size, out + latent_out)
    self.out = out

  def forward(self, x, state=None):
    batches = x.shape[:-1]
    init = x.reshape(-1, x.shape[-1])

    if state is None:
      state = torch.zeros(*init.shape[:-1], self.state_size, device=x.device)

    out = []
    for l in self.layers:
      state = l(init, state)
      out.append(state[..., :self.out])
    last = self.last(state)
    out.append(last[..., :self.out])
    return torch.cat(out, dim=-1).reshape(*batches, -1), \
      last[..., self.out:].reshape(*batches, -1)

class Upsampler(nn.Module):
  def __init__(
    self,

    in_size: int,
    out: int,

    kernel_size:int = 3,

    repeat:int = 6,
    in_features:int = 3,
    out_features:int = 3,
    feat_decay: float = 2,
    activation = nn.LeakyReLU(inplace=True),
  ):
    super().__init__()
    step_size = (out - in_size)//repeat
    self.sizes = list(range(in_size + step_size, out+step_size, step_size))
    self.sizes = self.sizes[:repeat]
    self.sizes[-1] = out
    assert(kernel_size % 2 == 1), "Must provide odd kernel upsampling"

    feat_sizes = [
      max(out_features, int(in_features // (feat_decay**i))) for i in range(repeat+1)
    ]

    self.base = nn.Conv2d(in_features, out_features, kernel_size, 1, (kernel_size-1)//2)

    self.convs = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(fs, nfs, kernel_size, 1, (kernel_size-1)//2),
        nn.Dropout2d(0.1, inplace=True),
        nn.LeakyReLU(inplace=True)
      )
      for fs, nfs in zip(feat_sizes, feat_sizes[1:])
      # Move from one feature size to the next
    ])

    self.combine = nn.ModuleList([
      nn.Conv2d(feat_sizes, out_features, kernel_size, 1, (kernel_size-1)//2)
      for feat_sizes in feat_sizes[1:]
    ])

    self.rgb_up_kind="bilinear"
    self.feat_up_kind="nearest"

  def forward(self, x):
    curr = x.permute(0,3,1,2)
    upscaled = self.base(curr) # (N, H_in, W_in, 3)

    for s, conv, combine in zip(self.sizes, self.convs, self.combine):
      resized_old=F.interpolate(upscaled,size=(s,s),mode=self.rgb_up_kind,align_corners=False)

      curr = conv(F.interpolate(curr, size=(s, s), mode=self.feat_up_kind))
      upscaled = resized_old + combine(curr)
    return upscaled.permute(0,2,3,1)

# An update operator from https://github.com/princeton-vl/RAFT/blob/master/core/update.py
# takes as input a vector, with some hidden state and returns iterative updates on it.
class UpdateOperator(nn.Module):
  def __init__(
    self,
    in_size: int = 3,
    out_size: Optional[int] = None,
    hidden_size: int = 32,
    iters: int = 3,
  ):
    super().__init__()
    self.hidden_size = hs = hidden_size
    self.convz = nn.Conv3d(hs+in_size, hs, kernel_size=3, padding=1)
    self.convr = nn.Conv3d(hs+in_size, hs, kernel_size=3, padding=1)
    self.convq = nn.Conv3d(hs+in_size, hs, kernel_size=3, padding=1)

    self.conv1 = nn.Conv3d(hs, hs, 3, padding=1)
    self.conv2 = nn.Conv3d(hs, in_size, 3, padding=1)
    self.relu = nn.LeakyReLU(inplace=True)

    assert(iters > 0), "Must have at least one iteration"
    self.iters = iters
    self.out_size = out_size or in_size

  def forward(self, x, h=None):
    x = x.permute(1,4,0,2,3)
    if h is None:
      shape = list(x.shape)
      shape[1] = self.hidden_size
      h = torch.zeros(shape, device=x.device, dtype=x.dtype)
    else: raise NotImplementedError("TODO Have to permute h")
    init_x = x
    # total change in x
    for i in range(self.iters):
      hx = torch.cat([x.detach(), h], dim=1)
      z = self.convz(hx).sigmoid()
      r = self.convr(hx).sigmoid()
      q = self.convq(torch.cat([r*h, x], dim=1)).sigmoid()

      h = (1-z) * h + z * q
      dx = self.conv2(self.relu(self.conv1(h)))
      x = x + dx
    # returns total delta, rather than the shifted input
    out = (x - init_x).permute(2,0,3,4,1)[..., :self.out_size]
    return out


class SpatialEncoder(nn.Module):
  # Encodes an image into a latent vector, for use in PixelNeRF
  def __init__(
    self,
    latent_size: int =64,
    num_layers: int = 4,
  ):
    super().__init__()
    self.latent = None
    self.model = torchvision.models.resnet34(pretrained=True).eval()
    self.latent_size = latent_size
    self.num_layers = num_layers
  # (B, C = 3, H, W) -> (B, L, H, W)
  def forward(self, img):
    img = img.permute(0,3,1,2)
    l_sz = img.shape[2:]
    x = self.model.conv1(img)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    latents = [x]
    # TODO other latent layers?

    ls = [F.interpolate(l, l_sz, mode="bilinear", align_corners=True) for l in latents]
    # necessary to detach here because we don't gradients to resnet
    # TODO maybe want latent gradients? Have to experiment with it.
    self.latents = torch.cat(latents, dim=1).detach().requires_grad_(False)
    return self.latents
  def sample(self, uvs: torch.Tensor, mode="bilinear", align_corners=True):
    assert(self.latents is not None), "Expected latents to be assigned in encoder"
    latents = F.grid_sample(
      self.latents,
      uvs.unsqueeze(0),
      mode=mode,
      align_corners=align_corners,
    )
    return latents.permute(0,2,3,1)

class Discriminator(nn.Module):
  def __init__(
    self,
    in_size=128,
    in_channels=3,
    hidden_size=64,
  ):
    super().__init__()
    self.in_size=128,
    hs = hidden_size
    ins = in_size
    assert(ins % 32 == 0)
    self.main = nn.Sequential(
      # input is 3 x ins x ins
      nn.Conv2d(in_channels, hidden_size, 4, 2, 1, bias=False),
      nn.LeakyReLU(inplace=True),
      # state size. (hs) x ins/2 x ins/2
      nn.Conv2d(hs, hs * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(hs * 2),
      nn.LeakyReLU(inplace=True),
      # state size. (hs*2) x ins/4 x ins/4
      nn.Conv2d(hs * 2, hs * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(hs * 4),
      nn.LeakyReLU(inplace=True),
      # state size. (hs*4) x ins/8 x ins/8
      nn.Conv2d(hs * 4, hs * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(hs * 8),
      nn.LeakyReLU(inplace=True),
      # state size. (hs*8) x ins/16 x ins/16
      nn.Conv2d(hs * 8, hs * 16, 4, 2, 1, bias=False),
    )
    self.last = nn.Linear(hs * ins * ins / 64, 1)
  def forward(self, x, ref):
    fake = self.single_image_loss(x)
    real = self.single_image_loss(ref)
    raise NotImplementedError()
  def single_image_loss(self, x):
    # x: Batch, H, W, 3
    x = TVF.resize(x.permute(0, 3, 1, 2), size=(self.in_size, self.in_size))

    # out: Batch, 1
    return self.last(self.main(x))


def gram_matrix(img):
  batch, c, w, h = img.size()
  # a=batch size(=1)
  # b=number of feature maps
  # (c,d)=dimensions of a f. map (N=c*d)

  features = img.reshape(batch * c, w * h)  # resise F_XL into \hat F_XL

  G = torch.mm(features, features.t())  # compute the gram product

  # we 'normalize' the values of the gram matrix
  # by dividing by the number of element in each feature maps.
  return G.div(batch * c * w * h)

# performs N class classification of some set of points along with feature vectors.
class PointNet(nn.Module):
  def __init__(
    self,
    feature_size:int=7,
    classes:int=2,
    enc=None,
    intermediate_size=128,
  ):
    super().__init__()
    feats=feature_size
    self.first = SkipConnMLP(
      in_size=feats, out=intermediate_size,
      enc=enc, init="xavier",
      num_layers=3, skip=999
    )
    self.second = SkipConnMLP(
      in_size=intermediate_size * 2, out=classes,
      init="xavier", num_layers=3, skip=999
    )
  def forward(self, pos, feats):
    # input has shape (batches, num_samples, C)
    first = self.first(torch.cat([pos, feats], dim=-1))
    # smooth min & smooth max over items
    first_pool = -torch.logsumexp(-first, dim=-2)
    second_pool = torch.logsumexp(first, dim=-2)
    pooled = torch.cat([first_pool, second_pool], dim=-1)
    return self.second(pooled)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        norm_mean = torch.tensor([0.485, 0.456, 0.406])
        norm_std = torch.tensor([0.229, 0.224, 0.225])
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = nn.Parameter(norm_mean.reshape(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(norm_std.reshape(-1, 1, 1), requires_grad=False)
    def forward(self, img):
      return (img - self.mean) / self.std

class StyleLoss(nn.Module):
  def __init__(self, feats):
    super().__init__()
    self.target = nn.Parameter(gram_matrix(feats).detach(), requires_grad=False)
  def forward(self, x):
    self.loss = F.mse_loss(gram_matrix(x), self.target)
    return x

class ContentLoss(nn.Module):
  def __init__(self, feats):
    super().__init__()
    self.feats = nn.Parameter(feats.detach(), requires_grad=False)
  def forward(self, x):
    self.loss = F.mse_loss(self.feats, x)
    return x

class StyleTransfer(nn.Module):
  def __init__(self, style_img, content_img=None):
    super().__init__()
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_layers = ['conv_4']
    self.norm = Normalization()
    self.cnn = models.vgg19(pretrained=True).features.eval()
    # An ordered model of all the style and content losses
    model = nn.Sequential(self.norm)

    # list of a bunch of style loss modules
    style_losses = []
    content_losses = []

    i = 0
    for layer in self.cnn.children():
      if isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv_{}'.format(i)
      elif isinstance(layer, nn.ReLU):
        name = 'relu_{}'.format(i)
        # The in-place version doesn't play very nicely with the ContentLoss
        # and StyleLoss we insert below. So we replace with out-of-place
        # ones here.
        layer = nn.ReLU(inplace=False)
      elif isinstance(layer, nn.MaxPool2d): name = 'pool_{}'.format(i)
      elif isinstance(layer, nn.BatchNorm2d): name = 'bn_{}'.format(i)
      else: raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

      model.add_module(name, layer)

      if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)
      if name in content_layers and content_img is not None:
        target_feature = model(content_img).detach()
        content_loss = ContentLoss(target_feature)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
      if isinstance(model[i], StyleLoss): break
      if isinstance(model[i], ContentLoss): break

    self.model = model[:(i + 1)]
    self.style_losses = nn.ModuleList(style_losses)
    self.content_losses = nn.ModuleList(content_losses)

  def forward(self, x):
    self.model(x)
    sl_loss = 0
    cl_loss = 0
    for sl in self.style_losses: sl_loss += sl.loss
    for cl in self.content_losses: cl_loss += cl.loss
    return sl_loss, cl_loss

# https://arxiv.org/pdf/1903.09410.pdf
# Trying to parse hard to understand stuff.
class MonteCarloBNLinear(nn.Module):
  def __init__(self, in_features, out_features, bias=True, monte_carlo_samples:int=30):
    self.linear = nn.Linear(in_features, out_features,bias=bias);
    self.mc_samples = monte_carlo_samples
  def forward(self, x):
    if not self.training: return self.linear(x)
    x = x.expand(self.mc_samples, *x.shape)
    out = self.layers(x.reshape(-1, *x.shape[1:]))
    # training=True only checks that some parameters are correct but doesn't modify output
    out = F.batch_norm(out, torch.randn_like(out), torch.randn_like(out), training=True)
    out = out.reshape(self.mc_samples, *x.shape)
    self._var = out.stddev(dim=0)
    return out.mean(dim=0)
  def var(shape=None):
    if shape is None: return self.var
    return self.var.reshape_as(shape)




