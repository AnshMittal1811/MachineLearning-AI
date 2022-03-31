import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from .nerf import ( CommonNeRF, compute_pts_ts )
from .neural_blocks import ( SkipConnMLP, FourierEncoder, NNEncoder )
from .utils import ( autograd, smooth_min, curl_divergence, elev_azim_to_dir )
import src.refl as refl
import src.march as march
import src.renderers as renderers
from tqdm import trange

def load(args, with_integrator:bool):
  cons = sdf_kinds.get(args.sdf_kind, None)
  if cons is None: raise NotImplementedError(f"Unknown SDF kind: {args.sdf_kind}")

  model = cons(intermediate_size=args.shape_to_refl_size)

  if args.sphere_init: model.set_to_sphere()

  if args.bound_sphere_rad > 0: model = UnitSphere(inner=model,rad=args.bound_sphere_rad)
  # refl inst may also have a nested light
  refl_inst = refl.load(args, args.refl_kind, args.space_kind, model.intermediate_size)
  isect = march.load_intersection_kind(args.sdf_isect_kind)

  sdf = SDF(model, refl_inst, isect=isect, t_near=args.near, t_far=args.far)
  if args.integrator_kind is not None and with_integrator:
    return renderers.load(args, sdf, refl_inst)

  return sdf

class SDFModel(nn.Module):
  def __init__(
    self,
    intermediate_size: int = 32,
  ):
    super().__init__()
    self.intermediate_size = intermediate_size
  def forward(self, _pts): raise NotImplementedError()

  def normals(self, pts, values = None):
    with torch.enable_grad():
      autograd_pts = pts if pts.requires_grad else pts.requires_grad_()

      if values is None: values = self(autograd_pts)
      normals = autograd(autograd_pts, values)
    return normals
  # will optimize this SDF to be a sphere at the start
  def set_to_sphere(self, rad:float = 0.5, iters:int=1000):
    opt = optim.Adam(self.parameters(), lr=5e-5, weight_decay=0)
    t = trange(iters)
    for i in t:
      opt.zero_grad()
      v = 4*torch.randn(5000, 3)
      got = self(v)[...,0]
      exp = v.norm(dim=-1) - rad
      loss = F.mse_loss(got, exp)
      t.set_postfix(l=loss.item())
      loss.backward()
      opt.step()

# Wraps another SDF as the intersection of a sphere centered at the origin.
class UnitSphere(SDFModel):
  def __init__(
    self,
    inner: SDFModel,
    rad:float=3,
  ):
    super().__init__(intermediate_size=inner.intermediate_size)
    self.inner = inner
    self.rad = rad

  def forward(self, pts):
    sph = torch.linalg.norm(pts, dim=-1, ord=2) - self.rad
    inner = self.inner(pts)
    return torch.cat([
      torch.maximum(inner[..., 0], sph).unsqueeze(-1),
      inner[..., 1:],
    ], dim=-1)

class SDF(nn.Module):
  def __init__(
    self,
    underlying: SDFModel,
    reflectance: refl.Reflectance,
    isect,
    t_near: float,
    t_far: float,
    alpha:int = 1000,
  ):
    super().__init__()
    assert(isinstance(underlying, SDFModel))
    self.underlying = underlying
    self.refl = reflectance
    self.far = t_far
    self.near = t_near
    self.alpha = alpha
    self.isect=isect

  @property
  def sdf(self): return self

  @property
  def intermediate_size(self): return self.underlying.intermediate_size

  def normals(self, pts, values = None): return self.underlying.normals(pts, values)
  def from_pts(self, pts):
    raw = self.underlying(pts)
    latent = raw[..., 1:]
    return raw[..., 0], latent if latent.shape[-1] != 0 else None

  def intersect_w_n(self, r_o, r_d):
    pts, hit, t, tput = self.isect(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      eps=5e-5, iters=128 if self.training else 256,
    )
    if self.training:
      if tput is None: tput = self.throughput(r_o, r_d)
      else: tput = -self.alpha * tput
    return pts, hit, tput, self.normals(pts)
  def intersect_mask(self, r_o, r_d, near=None, far=None, eps=1e-3):
    with torch.no_grad():
      throughput, _, _, _ = march.throughput_with_sign_change(
        self.underlying, r_o, r_d,
        near=self.near if near is None else near,
        far=self.far if far is None else far,
        # since this is just for intersection, alright to use fewer steps
        batch_size=32 if self.training else 196,
      )
      hits = throughput < eps
      return ~hits, throughput, None
  def forward(self, rays, with_throughput=True):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t, tput = self.isect(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=128 if self.training else 192,
    )
    latent = None if self.intermediate_size == 0 else self.underlying(pts[hit])[..., 1:]
    out = torch.zeros_like(r_d)
    n = None
    if self.refl.can_use_normal:
      self.n = torch.zeros_like(out)
      n = self.normals(pts[hit])
      self.n[hit] = n
    # use masking in order to speed up efficiency
    out[hit] = self.refl(
      x=pts[hit], view=r_d[hit], normal=n,
      latent=latent, mask=hit,
    )
    if with_throughput and self.training:
      if tput is None: tput = self.throughput(r_o, r_d)
      else: tput = -self.alpha * tput
      out = torch.cat([out, tput], dim=-1)
    return out
  def debug_normals(self, rays):
    r_o, r_d = rays.split([3,3], dim=-1)
    pts, hit, t = self.isect(
      self.underlying, r_o, r_d, near=self.near, far=self.far,
      iters=128 if self.training else 192,
    )
    latent = None if self.underlying.intermediate_size == 0 else self.underlying(pts[hit])[..., 1:]
    out = torch.zeros_like(r_d)
    out[hit] = self.normals(pts[hit])
    return out
  def throughput(self, r_o, r_d):
    tput, _best_pos = march.throughput(self.underlying, r_o, r_d, self.near, self.far)
    return -self.alpha*tput.unsqueeze(-1)


class SmoothedSpheres(SDFModel):
  def __init__(
    self,
    n:int=128,

    with_mlp=True,
    **kwargs,
  ):
    super().__init__(**kwargs)
    # has no latent size
    self.intermediate_size = 0

    self.centers = nn.Parameter(0.3 * torch.rand(n,3, requires_grad=True) - 0.15)
    self.radii = nn.Parameter(0.2 * torch.rand(n, requires_grad=True) - 0.1)

    self.tfs = nn.Parameter(torch.zeros(n, 3, 3, requires_grad=True))
    if with_mlp:
      self.mlp = SkipConnMLP(
        in_size=3, out=1, num_layers=5, hidden_size=128,
        enc=FourierEncoder(input_dims=3), init="xavier",
      )

  @torch.jit.export
  def transform(self, p):
    tfs = self.tfs + torch.eye(3, device=p.device).unsqueeze(0)
    return torch.einsum("ijk,ibk->ibj", tfs, p.expand(tfs.shape[0], -1, -1))

  def forward(self, p):
    q = self.transform(p.reshape(-1, 3).unsqueeze(0)) - self.centers.unsqueeze(1)
    sd = q.norm(p=2, dim=-1) - self.radii.unsqueeze(-1)
    out = smooth_min(sd, k=32.).reshape(p.shape[:-1] + (1,))
    if hasattr(self, "mlp"): out = out + self.mlp(p).tanh() * (1-out.sigmoid())
    return out


def dot(a,b, dim:int=-1, keepdim:bool=False): return (a * b).sum(dim=dim,keepdim=keepdim)
# dot2(a) = dot(a,a)
def dot2(a, dim:int=-1, keepdim:bool=False): return dot(a,a,dim=dim,keepdim=keepdim)

# triangle is similar to spheres but contains triangles instead
class Triangles(SDFModel):
  def __init__(
    self,
    n:int=32,

    **kwargs,
  ):
    super().__init__(**kwargs)
    # has no latent size
    self.intermediate_size = 0

    # each triangle requires 3 points
    self.points = nn.Parameter(0.3 * torch.rand(n,3,3, requires_grad=True) - 0.15)
    #self.thickness = nn.Parameter(0.3 * torch.rand(n, requires_grad=True) - 0.15)

  def forward(self, p):
    pa,pb,pc = (p.reshape(-1, 1, 1, 3) - self.points).split([1,1,1],dim=-2)
    ac,ba,cb = (self.points - self.points.roll(1, dims=-2)).split([1,1,1], dim=-2)
    nor = torch.cross(ba, ac, dim=-1)

    sidedness = \
      dot(torch.cross(ba, nor), pa, dim=-1).sign() + \
      dot(torch.cross(cb, nor), pb, dim=-1).sign() + \
      dot(torch.cross(ac, nor), pc, dim=-1).sign()


    same_sided = dot2(ba*(dot(ba, pa,keepdim=True)/dot2(ba,keepdim=True)).clamp(min=0,max=1)-pa)\
        .minimum(dot2(cb*(dot(cb, pb,keepdim=True)/dot2(cb,keepdim=True)).clamp(min=0,max=1)-pb))\
        .minimum(dot2(ac*(dot(ac, pc,keepdim=True)/dot2(ac,keepdim=True)).clamp(min=0,max=1)-pc))

    opp_sided = dot(nor, pa,dim=-1).square()/dot(nor,nor, dim=-1)
    out = torch.where(sidedness < 2, same_sided, opp_sided).clamp(min=1e-8).sqrt()
    # need to add a smooth min across all the triangles as well as an extrusion
    # apply thickness to each triangle to allow certain ones to take up more space.
    out = out.squeeze(-1) - 4e-2
    # smooth min or just normal union?
    return smooth_min(out,dim=-1).reshape(p.shape[:-1] + (1,))

class MLP(SDFModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mlp = SkipConnMLP(
      in_size=3, out=1+self.intermediate_size,
      enc=FourierEncoder(input_dims=3, sigma=1<<4),
      num_layers=6, hidden_size=256, init="xavier",
    )
  def forward(self, x): return self.mlp(x)

# CurlMLP is an irrotational field that produces an SDF.
# It can be thought of as generating a Signed Directional Distance Function,
# Which is dF(x)/dx * sign(F(x)), but the sign is approximated by tanh.
# This form works because SDFs have Curl(SDF) = 0 almost everywhere.
class CurlMLP(SDFModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mlp = SkipConnMLP(
      in_size=3, out=1+self.intermediate_size, enc=FourierEncoder(input_dims=3, sigma=1<<5),
      num_layers=6, hidden_size=256, init="xavier",
    )
  def forward(self, x):
    with torch.enable_grad():
      x = x if x.requires_grad else x.requires_grad_()
      field, latent = self.mlp(x).split([1, self.intermediate_size], dim=-1)
      field = torch.linalg.norm(autograd(x, field), dim=-1, keepdim=True) * field.tanh()
      return torch.cat([field, latent], dim=-1)

class SIREN(SDFModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.siren = SkipConnMLP(
      in_size=3, out=1+self.intermediate_size,
      num_layers=5, hidden_size=256,
      activation=torch.sin,
      skip=3, init="siren",
    )
  def forward(self, x): return self.siren(x)

# TODO verify this works? Haven't tried it out a ton on a lot of methods
class Local(SDFModel):
  def __init__(
    self,
    partition_sz: int = 0.5,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.part_sz = partition_sz
    self.latent = SkipConnMLP(in_size=3,out=self.intermediate_size,skip=4)
    self.tform = SkipConnMLP(
      in_size=3, out=1+self.intermediate_size, latent_size=self.intermediate_size,
      enc=NNEncoder(input_dims=3),
    )
  def forward(self, x):
    local = x % self.part_sz
    latent = self.latent(x/self.part_sz)
    return self.tform(local, latent)

sdf_kinds = {
  "mlp": MLP,
  "siren": SIREN,
  "local": Local,
  "curl-mlp": CurlMLP,
  # Classical models which don't work that well
  "spheres": SmoothedSpheres,
  "triangles": Triangles,
}
