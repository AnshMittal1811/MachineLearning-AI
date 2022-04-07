import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from .neural_blocks import ( SkipConnMLP, NNEncoder, FourierEncoder )
from .utils import ( autograd, eikonal_loss, dir_to_elev_azim, upshifted_sigmoid )
from .refl import ( LightAndRefl )

def load(args, shape, light_and_refl: LightAndRefl):
  assert(isinstance(light_and_refl, LightAndRefl)), "Need light and reflectance for integrator"

  if args.integrator_kind is None: return None
  elif args.integrator_kind == "direct": cons = Direct
  elif args.integrator_kind == "path": cons = Path
  else: raise NotImplementedError(f"load integrator: {args.integrator_kind}")
  ls = 0
  if hasattr(shape, "latent_size"): ls = shape.latent_size
  elif hasattr(shape, "total_latent_size"): ls = shape.total_latent_size()


  occ = load_occlusion_kind(args, args.occ_kind, ls)
  integ = cons(shape=shape, refl=light_and_refl.refl, occlusion=occ)

  return integ

# no shadow
def lighting_wo_isect(pts, lights, isect_fn, latent=None, mask=None):
  dir, _, spectrum = lights(pts if mask is None else pts[mask], mask=mask)
  return dir, spectrum

# hard shadow lighting
class LightingWIsect(nn.Module):
  def __init__(self, latent_size:int): super().__init__()
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    visible, _, _ = isect_fn(pts, dir, near=0.1, far=far)
    spectrum = torch.where(
      visible[...,None],
      spectrum,
      torch.zeros_like(spectrum)
    )
    return dir, spectrum

class LearnedLighting(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    in_size=5
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size, num_layers=5, hidden_size=128,
      enc=FourierEncoder(input_dims=in_size), init="xavier",
    )
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask.any() else 6
    # TODO why doesn't this isect fn seem to work?
    visible, _, _ = isect_fn(r_o=pts, r_d=dir, near=2e-3, far=far, eps=1e-3)
    elaz = dir_to_elev_azim(dir)
    att = self.attenuation(torch.cat([pts, elaz], dim=-1), latent).sigmoid()
    spectrum = torch.where(visible.reshape_as(att), spectrum, spectrum * att)
    return dir, spectrum

class LearnedConstantSoftLighting(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
  ):
    super().__init__()
    in_size=5
    self.alpha = nn.Parameter(torch.tensor(0., requires_grad=True), requires_grad=True)
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if mask and mask.any() else 6
    visible, _, _ = isect_fn(r_o=pts, r_d=dir, near=1e-2, far=far, eps=1e-3)
    hit_att = visible + (~visible) * self.alpha.sigmoid()
    return dir, spectrum * hit_att.unsqueeze(-1)

def just_pos(pos, dir): return pos
def pos_elaz(pos, dir): return torch.cat([pos, dir_to_elev_azim(dir)], dim=-1)

all_learned_occ_kinds = {
  "pos": (just_pos, 3),
  "pos-elaz": (pos_elaz, 5)
}

# Can we consider this as a kind of learned ambient occlusion?
class AllLearnedOcc(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
    # pass both elaz and direction
    kind="pos",
  ):
    super().__init__()
    self.component_fn, in_size = all_learned_occ_kinds[kind]
    self.attenuation = SkipConnMLP(
      in_size=in_size, out=1, latent_size=latent_size,
      enc=FourierEncoder(input_dims=in_size),
      num_layers=6, hidden_size=256, init="xavier",
    )
  @property
  def all_learned_occ(self): return self
  def encode(self, pts, dir, latent):
    self.raw_att = self.attenuation(self.component_fn(pts, dir), latent)
    return upshifted_sigmoid(self.raw_att)
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    pts = pts if mask is None else pts[mask]
    dir, _, spectrum = lights(pts, mask=mask)
    return dir, spectrum * self.encode(pts, dir, latent)

class JointLearnedConstOcc(nn.Module):
  def __init__(
    self,
    latent_size:int=0,
    alo: AllLearnedOcc = None,
    lcsl: LearnedConstantSoftLighting=None,
  ):
    if alo is None: alo = AllLearnedOcc(latent_size=latent_size)
    assert(isinstance(alo, AllLearnedOcc)), "Must pass an instance of AllLearnedOcc"
    if lcsl is None: lcsl = LearnedConstantSoftLighting(latent_size=latent_size)
    assert(isinstance(lcsl, LearnedConstantSoftLighting)), "Must pass an instance of AllLearnedOcc"
    super().__init__()
    self.alo = alo
    self.lcsl = lcsl
  @property
  def all_learned_occ(self): return self.alo
  def forward(self, pts, lights, isect_fn, latent=None, mask=None):
    if mask is not None: raise NotImplementedError("TODO did not implement handling mask")
    dir, dist, spectrum = lights(pts, mask=mask)
    far = dist.max().item() if isinstance(dist, torch.Tensor) else dist
    # only include the all learned occ if training.
    all_att = self.alo.encode(pts, dir, latent)
    visible, _, _ = isect_fn(r_o=pts, r_d=dir, near=1e-1, far=far, eps=1e-3)
    hit_att = visible + (~visible) * self.lcsl.alpha.sigmoid()
    spectrum = spectrum * all_att * hit_att.unsqueeze(-1)
    return dir, spectrum

occ_kinds = {
  None: lambda **kwargs: lighting_wo_isect,
  "hard": LightingWIsect,
  "learned": LearnedLighting,
  "learned-const": LearnedConstantSoftLighting,
  "all-learned": AllLearnedOcc,
  "joint-all-const": JointLearnedConstOcc,
}

def load_occlusion_kind(args, kind=None, latent_size:int=0):
  con = occ_kinds.get(kind, -1)
  if con == -1: raise NotImplementedError(f"load occlusion: {args.occ_kind}")
  kwargs = { "latent_size": latent_size, }
  if kind == "all-learned":
    try: kwargs["kind"] = args.all_learned_occ_kind
    except: kwargs["kind"] = "pos-elaz"

  return con(**kwargs)

class Renderer(nn.Module):
  def __init__(
    self, shape, refl,
    occlusion,
  ):
    super().__init__()
    self.shape = shape
    self.refl = refl
    self.occ = occlusion

  def forward(self, _rays): raise NotImplementedError()

class Direct(Renderer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  @property
  def sdf(self): return self.shape
  def total_latent_size(self): return self.shape.latent_size
  def set_refl(self, refl): self.refl = refl
  def forward(s, rays): return direct(s.shape, s.refl, s.occ, rays, s.training)

# Functional version of integration
def direct(shape, refl, occ, rays, training=True):
  r_o, r_d = rays.split([3, 3], dim=-1)

  pts, hits, tput, n = shape.intersect_w_n(r_o, r_d)
  _, latent = shape.from_pts(pts[hits])

  out = torch.zeros_like(r_d)
  for light in refl.light.iter():
    light_dir, light_val = occ(pts, light, shape.intersect_mask, mask=hits, latent=latent)
    bsdf_val = refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
    out[hits] = out[hits] + bsdf_val * light_val
  if training: out = torch.cat([out, tput], dim=-1)
  return out

def path(shape, refl, occ, rays, training=True):
  r_o, r_d = rays.split([3, 3], dim=-1)

  pts, hits, tput, n = shape.intersect_w_n(r_o, r_d)
  _, latent = shape.from_pts(pts[hits])

  out = torch.zeros_like(r_d)
  for light in refl.light.iter():
    light_dir, light_val = occ(pts, light, shape.intersect_mask, mask=hits, latent=latent)
    bsdf_val = refl(x=pts[hits], view=r_d[hits], normal=n[hits], light=light_dir, latent=latent)
    out[hits] = out[hits] + bsdf_val * light_val

  # TODO this should just be a random sample of pts in some range?
  pts_2nd_ord = pts.reshape(-1, 3)
  pts_2nd_ord = pts[torch.randint(high=pts_2nd_ord.shape[0], size=32, device=pts.device), :]
  with torch.no_grad():
    # compute light to set of points
    light_dir, light_val = occ(pts_2nd_ord, shape.intersect_mask, latent=latent)
    # compute dir from each of the 2nd order pts to the main points
    dirs = pts_2nd_ord - pts
    # TODO apply the learned occlusion here
    att = occ.attenuation(torch.cat([pts_2nd_ord, dirs], dim=-1), latent=latent)
    # TODO this should account for the BSDF when computing the reflected component
    out[hits] = out[hits] + att * light_val
  if training: out = torch.cat([out, tput], dim=-1)
  return out

class Path(Renderer):
  def __init__(
    self,
    bounces:int=1,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.bounces = bounces
  def forward(self, rays):
    raise NotImplementedError()
