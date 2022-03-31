import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import Optional

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import (
  coordinate_system,
  autograd, eikonal_loss, dir_to_elev_azim, rotate_vector, load_sigmoid
)
import src.lights as lights
from .spherical_harmonics import eval_sh


def load(args, refl_kind:str, space_kind:str, latent_size:int):
  if space_kind == "identity": space = IdentitySpace
  elif space_kind == "surface": space = SurfaceSpace
  elif space_kind == "none": space = NoSpace
  else: raise NotImplementedError()

  kwargs = {
    "latent_size": latent_size,
    "act": args.sigmoid_kind,
    "out_features": args.feature_space,
    "normal": args.normal_kind,
    "bidirectional": args.refl_bidirectional,
    "space": space(),
  }
  cons = refl_kinds.get(refl_kind, None)
  if cons is None: raise NotImplementedError(f"refl kind: {args.refl_kind}")
  if refl_kind == "basic":
    if args.light_kind is not None: kwargs["light"] = "elaz"
  elif refl_kind == "fourier": kwargs["order"] = args.refl_order
  elif refl_kind == "sph-har": kwargs["order"] = args.refl_order
  elif refl_kind == "weighted":
    subs = args.weighted_subrefl_kinds
    assert(len(subs) > 1), "Specifying one subrefl is pointless."
    kwargs["choices"] = [load(args, c, "none", latent_size) for c in subs]

  # TODO assign view, normal, lighting here?
  refl = cons(**kwargs)

  if args.light_kind is not None and refl.can_use_light:
    light = lights.load(args)
    refl = LightAndRefl(refl=refl,light=light)

  return refl

# A combo of an existing reflectance model and some light.
# It is easier than having them separate because the light is only necessary
# for the reflectance model.
class LightAndRefl(nn.Module):
  def __init__(self, refl, light):
    super().__init__()
    self.refl = refl
    self.light = light
    self.spectrum = None

  @property
  def can_use_normal(self): return self.refl.can_use_normal
  @property
  def latent_size(self): return self.refl.latent_size

  def forward(self, x, view=None, normal=None, light=None, latent=None, mask=None):
    if light is None: light = self.light(x)[0]
    return self.refl(x, view, normal, light, latent)

# Convert from arbitrary 3d space to a 2d encoding.
class SurfaceSpace(nn.Module):
  def __init__(
    self,
    final_act=nn.Identity(),
  ):
    super().__init__()
    self.encode = SkipConnMLP(
      in_size=3, out=2, activation=act, enc=FourierEncoder(input_dims=3), num_layers=5, hidden_size=128,
    )
    self.act = activation
  def forward(self, x): return self.act(self.encode(x))

  @property
  def dims(self): return 2

# Use raw 3d point as value for encoding
class IdentitySpace(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x): return x
  @property
  def dims(self): return 3

# Do not encode the space whatsoever, only relying on other view-dependent features.
class NoSpace(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x): torch.empty((*x.shape[:-1], 0), device=x.device, dtype=torch.float)
  @property
  def dims(self): return 0

class Reflectance(nn.Module):
  def __init__(
    self,
    act="thin",
    latent_size:int = 0,
    out_features:int = 3,
    bidirectional: bool = True,

    # These exist to delete unused parameters, but don't use kwargs so if anything falls through
    # it will be easy to spot.
    normal=None,
    light=None,
  ):
    super().__init__()
    self.latent_size = latent_size
    self.out_features = out_features
    self.bidirectional = bidirectional

    self.act = load_sigmoid(act)

  def forward(self, x, view,normal=None,light=None,latent=None): raise NotImplementedError()
  @property
  def can_use_normal(self): return False
  @property
  def can_use_light(self): return False

  # in order to use the reflectance model for voxels, convert it into a special form.
  # Implementors should return a tuple of number of parameters, and a function which
  # takes a Tensor[..., num_params] --> RGB.
  def to_voxel(self): raise NotImplementedError(f"{type(self)} does not have a voxel repr")

  # TODO allow any arbitrary reflectance model to predict spherical harmonic parameters then use
  # this.
  def sph_ham(sh_coeffs, view):
    # not spherical harmonic coefficients
    if self.order == 0: return sh_coeffs
    return eval_sh(
      self.order,
      sh_coeffs.reshape(sh_coeffs.shape[:-1] + (self.out_features, -1)),
      F.normalize(view, dim=-1),
    )

def ident(x): return F.normalize(x, dim=-1)
def empty(_): return None
# encode a direction either directly as a vector, as a theta & phi, or as nothing.
def enc_norm_dir(kind=None):
  if kind is None: return 0, empty
  elif kind == "raw": return 3, ident
  elif kind == "elaz": return 2, dir_to_elev_azim
  else: raise NotImplementedError(f"enc_norm_dir: {kind}")

# basic reflectance takes a position and a direction and other components
class Basic(Reflectance):
  def __init__(
    self,
    space=None,

    view="elaz",
    normal=None,
    light=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    view_dims, self.view_enc = enc_norm_dir(view)
    normal_dims, self.normal_enc = enc_norm_dir(normal)
    light_dims, self.light_enc = enc_norm_dir(light)
    in_size = view_dims + normal_dims + light_dims + space.dims
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
    self.space = space

  @property
  def can_use_normal(self): return self.normal_enc != empty
  @property
  def can_use_light(self): return self.light_enc != empty

  def forward(self,x,view,normal=None,light=None,latent=None):
    x = self.space(x)
    view = self.view_enc(view)
    normal = self.normal_enc(normal)
    self.light_enc = empty
    light = self.light_enc(light)
    v = torch.cat([v for v in [x, view, normal, light] if v is not None], dim=-1)
    return self.act(self.mlp(v, latent))

# view reflectance takes a view direction and a latent vector, and nothing else.
class View(Reflectance):
  def __init__(
    self,
    space=None,

    view="elaz",
    **kwargs,
  ):
    super().__init__(**kwargs)
    view_dims, self.view_enc = enc_norm_dir(view)
    in_size = view_dims+3
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      num_layers=4, hidden_size=256, init="siren", activation=torch.sin
    )
  def forward(self, x, view, normal=None, light=None, latent=None):
    v = self.view_enc(view)
    return self.act(self.mlp(torch.cat([x, v], dim=-1), latent))

# ViewLight reflectance takes a view direction, light and a latent vector.
class ViewLight(Reflectance):
  def __init__(self, space=None, view="elaz", light="elaz", **kwargs):
    super().__init__(**kwargs)
    view_dims, self.view_enc = enc_norm_dir(view)
    light_dims, self.light_enc = enc_norm_dir(light)
    in_size = view_dims + light_dims
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=256, init="xavier",
    )
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal=None, light=None, latent=None):
    v = self.view_enc(view)
    l = self.light_enc(light)
    return self.act(self.mlp(torch.cat([v, l], dim=-1), latent))

# Positional only (no view dependence)
class Positional(Reflectance):
  def __init__(self, space=None, **kwargs):
    super().__init__(**kwargs)
    self.mlp = SkipConnMLP(
      in_size=3, out=self.out_features, latent_size=self.latent_size,
      num_layers=5, hidden_size=256, init="siren", activation=torch.sin,
    )
  # Each voxel just requires the out_features which is usually RGB,
  # and does not need to do any special modifications.
  def voxel_forward(self, params, view): return self.act(params)
  def to_voxel(self):
    del self.mlp
    return self.out_features, self.voxel_forward
  def forward(self, x, view, normal=None, light=None, latent=None):
    return self.act(self.mlp(x, latent))


class PosLinearView(Reflectance):
  def __init__(self, space=None, view="raw", intermediate_size=64, **kwargs):
    super().__init__(**kwargs)
    view_size, self.view_enc = enc_norm_dir(view)
    self.im = intermediate_size
    self.pos = SkipConnMLP(
      in_size=3, out=self.out_features+self.im, latent_size=self.latent_size,
      num_layers=3, hidden_size=256, init="siren", activation=torch.sin,
    )
    self.view = SkipConnMLP(
      # linear shading, combined with specular highlight
      in_size=3+view_size, out=1, latent_size=self.latent_size + self.im,
      num_layers=3, hidden_size=128, init="siren", activation=torch.sin,
    )
  # Each voxel just requires the out_features which is usually RGB,
  # and does not need to do any special modifications.
  def pos_linear_voxel_forward(self, params, view):
    raw_rgb, sh_coeffs = params.split([self.out_features, self.num_sh_coeffs], dim=-1)
    linear_scale = eval_sh(
      self.order,
      sh_coeffs.reshape(*sh_coeffs.shape[:-1], 1, -1),
      F.normalize(view,dim=-1)
    ).sigmoid()
    return self.act(raw_rgb) * ((linear_scale/2)+0.5)
  def to_voxel(self):
    # explicitly delete the MLPs to save memory
    del self.pos
    del self.view

    self.order = order = 4 # TODO where to pass this into this function?
    self.num_sh_coeffs = (order+1) * (order+1)
    return self.out_features+self.num_sh_coeffs, self.pos_linear_voxel_forward

  def forward(self, x, view, normal=None, light=None, latent=None):
    pos, intermediate = self.act(self.pos(x, latent))\
      .split([self.out_features, self.im], dim=-1)
    view_latent = intermediate if latent is None else torch.cat([latent, intermediate], dim=-1)
    linear = self.view(torch.cat([x, self.view_enc(view)], dim=-1), view_latent).sigmoid()
    # constrain linear coefficient to 0.5-1.
    linear = (linear/2) + 0.5
    # We constrain gamma to the range 0.5 to 1.5, but could be larger.
    return linear * pos


class Diffuse(Reflectance):
  def __init__(
    self,
    space=None,

    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space

    in_size = space.dims
    self.diffuse_color = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      num_layers=3,hidden_size=512,
      enc=FourierEncoder(input_dims=in_size), init="xavier",
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    rgb = self.act(self.diffuse_color(self.space(x), latent))
    att = (normal * light).sum(dim=-1, keepdim=True)
    assert(((att <= 1.001) & (att >= -1.001)).all()), \
      f"{att.min().item()}, {att.max().item()}"
    if getattr(self, "bidirectional", True): att = att.maximum((-normal * light).sum(dim=-1, keepdim=True))
    # When clamping to 0, it only learns when directly illuminated which seems to be alright.
    return rgb * att

# https://pbr-book.org/3ed-2018/Reflection_Models/Fourier_Basis_BSDFs
class FourierBasis(Reflectance):
  def __init__(
    self,
    space=None,
    order:int =16,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space
    self.order = order

    in_size = space.dims
    self.fourier_coeffs = SkipConnMLP(
      in_size=in_size, out=self.order * self.out_features,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=6, hidden_size=128, init="xavier",
    )
    # this model does not use an activation function(?)
  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True
  @property
  def num_parameters(self): return self.order * self.out_features

  def forward(self, x, view, normal, light, latent=None):
    frame = coordinate_system(normal)
    wo = to_local(frame, F.normalize(view, dim=-1))
    wi = to_local(frame, light)
    #mu_i = -wi[..., 2]
    #mu_o = wo[..., 2]
    cos_phi = cos_D_phi(-wi, wo)
    cos_k_phis = [ torch.ones_like(cos_phi), cos_phi, ]
    for i in range(2, self.order):
      cos_k_phi = 2 * cos_phi * cos_k_phis[-1] - cos_k_phis[-2]
      cos_k_phis.append(cos_k_phi)
    cos_k_phis = torch.cat(cos_k_phis, dim=-1)
    fourier_coeffs = self.fourier_coeffs(x, latent)
    fourier_coeffs = fourier_coeffs.reshape(*x.shape[:-1], self.out_features, self.order)
    rgb = (fourier_coeffs * cos_k_phis.unsqueeze(-2)).sum(dim=-1)
    return rgb

# https://graphicscompendium.com/gamedev/15-pbr
# https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-31439-6_531
class CookTorrance(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space

    in_size = space.dims
    self.spec_frac = SkipConnMLP(
      in_size=in_size, out=1,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
    self.ior = SkipConnMLP(
      in_size=in_size, out=1,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
    self.facet_slope_dist = SkipConnMLP(
      in_size=in_size + 1, out=1,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
    self.diffuse_color = SkipConnMLP(
      in_size=in_size, out=3,
      latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    H = F.normalize(view + light, dim=-1) # Half vector between light and view
    n_dot_l = (normal * light).sum(dim=-1, keepdim=True)
    n_dot_v = (normal *  view).sum(dim=-1, keepdim=True)
    n_dot_h = (normal *     H).sum(dim=-1, keepdim=True)
    c = (view * H).sum(dim=-1, keepdim=True)

    # For now treat index of refraction in [1,3.5]
    ior = F.sigmoid(self.ior(x, latent)) * 2.5 + 1
    g_sq = ior * ior + c * c - 1
    g = g_sq.clamp(min=1e-8).sqrt()
    g_minus_c = g - c
    g_plus_c = g + c

    # TODO is below numerically stable?
    F = 0.5 * \
      g_minus_c.square()/g_plus_c.square().clamp(min=1e-8) * \
      (1 + (c * g_plus_c + 1).square()/(c * g_minus_c + 1).square().clamp(min=1e-8))

    G = torch.minimum(
      2 * n_dot_h * n_dot_v/c,
      2 * n_dot_h * n_dot_l/c,
    ).clamp(max=1)
    # TODO this should be normalized to 1 over the hemisphere?
    D = self.facet_slope_dist(torch.cat([x, n_dot_h], dim=-1), latent).sigmoid()
    r_s = F * D * G / (4 * n_dot_l * n_dot_v)

    r_d = self.diffuse_color(x, latent)

    spec_frac = self.spec_frac(x, latent).sigmoid()
    rgb = spec_frac * r_s + (1 - spec_frac) * r_d

    return rgb * n_dot_l

def cos_D_phi(wo, wi):
  wox,woy,woz = wo.split([1,1,1], dim=-1)
  wix,wiy,wiz = wi.split([1,1,1], dim=-1)
  cos_d_phi = (wox * wix + woy * wiy)/\
    torch.sqrt((wox*wox + woy*woy)*(wix*wix + wiy*wiy))
  return cos_d_phi.clamp(min=-1, max=1)

class WeightedChoice(Reflectance):
  def __init__(
    self,
    choices:[Reflectance],
    space=None,

    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    self.space = space

    for c in choices:
      assert(issubclass(type(c), Reflectance) or isinstance(c, LightAndRefl)), \
        f"Not refl: {type(c)}"

    self.choices = nn.ModuleList(choices)
    in_size = space.dims
    self.selection = SkipConnMLP(
      in_size=in_size, out=len(choices), latent_size=self.latent_size,
      init="xavier", enc=FourierEncoder(input_dims=in_size),
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    weights = self.selection(self.space(x), latent)
    weights = F.softmax(weights,dim=-1).unsqueeze(-2)
    subs = torch.stack([
      c(x, view, normal, light, latent) for c in self.choices
    ], dim=-1)
    return (weights * subs).sum(dim=-1)

class Rusin(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    if space is None: space = IdentitySpace()
    rusin_size = 3
    self.space = space
    in_size = rusin_size + space.dims
    self.rusin = SkipConnMLP(
      in_size=in_size, out=self.out_features, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size), init="xavier",

      num_layers=5, hidden_size=256,
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  # returns the raw results given rusin parameters
  def raw(self, rusin_params, latent=None):
    return self.act(self.rusin(rusin_params.cos(), latent))

  def forward(self, x, view, normal, light, latent=None):
    # NOTE detach the normals since there is no grounding of them w/ Rusin reflectance
    frame = coordinate_system(normal.detach())
    # have to move view and light into basis of normal
    wo = to_local(frame, F.normalize(view, dim=-1))
    wi = to_local(frame, light)
    rusin = rusin_params(wo, wi)
    params = torch.cat([rusin, self.space(x)], dim=-1)
    return self.act(self.rusin(params, latent))

# Helmholtz decomposition of the Rusin function.
# I believe this allows for separately learning the diffuse component
class RusinHelmholtz(Reflectance):
  def __init__(
    self,
    space=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    assert(out_features==3), "RusinHelmholtz can only output RGB"
    if space is None: space = IdentitySpace()
    rusin_size = 3
    self.space = space
    in_size = rusin_size + space.dims
    self.scalar_potential = SkipConnMLP(
      in_size=in_size, out=1, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size), init="xavier",
      num_layers=5, hidden_size=256,
    )
    self.solenoidal = SkipConnMLP(
      in_size=in_size, out=3, latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size), init="xavier",
      num_layers=5, hidden_size=256,
    )

  @property
  def can_use_normal(self): return True
  @property
  def can_use_light(self): return True

  def forward(self, x, view, normal, light, latent=None):
    # NOTE detach the normals since there is no grounding of them w/ Rusin reflectance
    frame = coordinate_system(normal.detach())
    # have to move view and light into basis of normal
    wo = to_local(frame, F.normalize(view, dim=-1))
    wi = to_local(frame, light)
    rusin = rusin_params(wo, wi).requires_grad_()
    pts = self.space(x)
    params = torch.cat([rusin, pts], dim=-1)
    scalar = self.scalar_potential(params, latent)
    irrotational = autograd(rusin, scalar)
    # TODO add in divergence-free component
    vector = self.solenoidal(params, latent)
    solenoidal = autograd(rusin, vector)
    print(out.shape)
    # TODO compute curl of solenoidal component.
    exit()
    return -irrotational


# The sum of an analytic and learned BRDF, intended to be the case that only one of them will
# have their parameters with gradients at a time so that optimizing them will guarantee the
# correctness of the SDF.
class AlternatingOptimization(nn.Module):
  def __init__(
    self,
    old_analytic=None,
    old_learned=None,
    **kwargs,
  ):
    super().__init__()
    # TODO possibly allow for different constructors
    self.analytic = old_analytic if old_analytic is not None else Diffuse(**kwargs)
    kwargs["act"] = "normal" # enforce the learned model is a sigmoid function.
    self.learned = old_learned if old_learned is not None else Rusin(**kwargs)
    self.add_blend()

    # always start by optimizing diffuse, else the learned rusinkiewicz is fixed
    self.learn_analytic = True
    for p in self.learned.parameters():
      p.requires_grad = False

  @property
  def can_use_normal(self): return self.analytic.can_use_normal or self.learned.can_use_normal
  @property
  def latent_size(self): return self.analytic.latent_size
  @property
  def can_use_light(self): return self.analytic.can_use_light or self.learned.can_use_light

  def toggle(self, learn_analytic:Optional[bool]=None):
    self.learn_analytic = not self.learn_analytic if learn_analytic is None else learn_analytic
    for p in self.analytic.parameters():
      p.requires_grad = self.learn_analytic
    for p in self.learned.parameters():
      p.requires_grad = not self.learn_analytic
  def add_blend(self):
    self.blend = SkipConnMLP(
      in_size=3, out=1, latent_size=self.analytic.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )

  def forward(self, x, view, normal, light, latent=None):
    learned = self.learned(x, view, normal,light, latent)
    analytic = self.analytic(x, view, normal, light, latent)
    if not hasattr(self, "blend"): self.add_blend()
    t = self.blend(x, latent).sigmoid()
    return t * learned + (1-t) * analytic

def nonzero_eps(v, eps: float=1e-7):
  # in theory should also be copysign of eps, but so small it doesn't matter
  # and torch.jit.script doesn't support it
  return torch.where(v.abs() < eps, torch.full_like(v, eps), v)

# assumes wo and wi are already in local coordinates
#@torch.jit.script
def rusin_params(wo, wi):
  wo = F.normalize(wo, eps=1e-6, dim=-1)
  wi = F.normalize(wi, eps=1e-6, dim=-1)
  e_1 = torch.tensor([0,1,0], device=wo.device, dtype=torch.float).expand_as(wo)
  e_2 = torch.tensor([0,0,1], device=wo.device, dtype=torch.float).expand_as(wo)

  H = F.normalize((wo + wi), eps=1e-6, dim=-1)

  cos_theta_h = H[..., 2]
  phi_h = torch.atan2(nonzero_eps(H[..., 1]), nonzero_eps(H[..., 0]))

  r = nonzero_eps(H[..., 1]).hypot(nonzero_eps(H[..., 0])).clamp(min=1e-6)
  c = (H[..., 0]/r).unsqueeze(-1)
  s = -(H[..., 1]/r).unsqueeze(-1)
  tmp = F.normalize(rotate_vector(wi, e_2, c, s), dim=-1)

  c = H[..., 2].unsqueeze(-1)
  s = -(1 - H[..., 2]).clamp(min=1e-6).sqrt().unsqueeze(-1)
  diff = F.normalize(rotate_vector(tmp, e_1, c, s), eps=1e-6, dim=-1)
  cos_theta_d = diff[..., 2]

  phi_d = torch.atan2(nonzero_eps(diff[..., 1]), nonzero_eps(diff[..., 0]))
  phi_d = phi_d.cos()
  #phi_d = torch.remainder(phi_d, math.pi)

  return torch.stack([phi_d, cos_theta_h, cos_theta_d], dim=-1)

# https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors
def coordinate_system2(n):
  n = F.normalize(n, eps=1e-6, dim=-1)
  x,y,z = n.split([1,1,1], dim=-1)
  s = torch.where(
    x.abs() > y.abs(),
    F.normalize(torch.cat([-z, torch.zeros_like(y), x], dim=-1), dim=-1),
    F.normalize(torch.cat([torch.zeros_like(x), z, -y], dim=-1), dim=-1),
  )
  t = torch.cross(n, s, dim=-1)
  return torch.stack([s,t,n], dim=-1)
# frame: [..., 3, 3], wo: [..., 3], return a vector of wo in the reference frame

def to_local(frame, wo):
  return F.normalize((frame * wo.unsqueeze(-1)).sum(dim=-2), eps=1e-7, dim=-1)

# Spherical Harmonics computes reflectance of a given viewing direction using the spherical
# harmonic basis.
class SphericalHarmonic(Reflectance):
  def __init__(
    self,
    space=None,
    order:int=2,
    view="elaz",

    **kwargs,
  ):
    super().__init__(**kwargs)
    assert(order >= 0 and order <= 4)
    in_size, self.view_enc = enc_norm_dir(view)
    self.order = order
    self.mlp = SkipConnMLP(
      in_size=in_size, out=self.out_features*((order+1)*(order+1)), latent_size=self.latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=5, hidden_size=128, init="xavier",
    )
  def to_voxel(self):
    order = self.order
    feats = self.out_features
    num_params = feats * (order+1) * (order+1)
    # Hopefully this allows the base class to be garbage collected.
    return num_params, lambda params, view: eval_sh(order,
      params.reshape(params.shape[:-1] + [feats,-1]),
      F.normalize(view, dim=-1),
    )
  def forward(self, x, view, normal=None, light=None, latent=None):
    v = self.view_enc(view)
    sh_coeffs = self.mlp(v, latent)
    rgb = eval_sh(
      self.order,
      sh_coeffs.reshape(sh_coeffs.shape[:-1] + (self.out_features, -1)),
      F.normalize(view, dim=-1),
    )
    return self.act(rgb)

refl_kinds = {
  "pos": Positional,
  "view":  View,
  "pos-linear-view": PosLinearView,
  "view-light": ViewLight,
  "basic": Basic,
  "diffuse": Diffuse,
  "rusin": Rusin,
  "rusin-helmholtz": RusinHelmholtz,
  # classical models with some order mechanism
  "sph-har": SphericalHarmonic,
  "fourier": FourierBasis,
  # meta refl models
  "weighted": WeightedChoice,
  # alternating optimiziation between diffuse and learned model
  # "alt-opt",
}
