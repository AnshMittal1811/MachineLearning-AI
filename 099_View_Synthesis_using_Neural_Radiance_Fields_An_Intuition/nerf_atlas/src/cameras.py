import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rotate_vector
from .neural_blocks import ( SkipConnMLP )
import random

# General Camera interface
class Camera(nn.Module):
  def __init__(self): super().__init__()
  # samples from positions in [0,size] screen space to global
  def sample_positions(self, positions): raise NotImplementedError()

# A camera made specifically for generating rays from NeRF models
class NeRFCamera(Camera):
  def __init__(
    self,
    cam_to_world:torch.tensor = None,
    focal: float=None,
    near: float = None,
    far: float = None,
  ):
    super().__init__()
    self.cam_to_world = nn.Parameter(cam_to_world, requires_grad=False)
    self.focal = focal
    self.near = near
    self.far = far

  def __len__(self): return self.cam_to_world.shape[0]

  @classmethod
  def identity(cls, batch_size: int):
    c2w = torch.tensor([
      [1,0,0, 0],
      [0,1,0, 0],
      [0,0,1, 0],
    ]).unsqueeze(0).expand(batch_size, 3, 4)
    return cls(cam_to_world=c2w, focal=0.5)

  # support indexing to get sub components of a camera
  def __getitem__(self, v):
    return NeRFCamera(cam_to_world=self.cam_to_world[v], focal=self.focal)

  def sample_positions(
    self,
    position_samples,
    size: int,
    with_noise=False,
  ):
    device=position_samples.device
    u,v = position_samples.split([1,1], dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack([
        (u - size * 0.5) / self.focal,
        -(v - size * 0.5) / self.focal,
        -torch.ones_like(u),
    ], dim=-1)
    r_d = torch.sum(d[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1)
    r_d = r_d.permute(2,0,1,3) # [H, W, B, 3] -> [B, H, W, 3]
    r_o = self.cam_to_world[..., :3, -1][:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)
  def project_pts(self, pts, size):
    untrans = pts - self.cam_to_world[..., :3, -1]
    w2c = torch.linalg.inv(self.cam_to_world[..., :3, :3])
    w = (w2c * pts[..., None, :]).sum(dim=-1)
    K = torch.tensor([self.focal, -self.focal, -1], device=pts.device)
    v = K[None] * w
    v = v[..., :-1]/v[..., -1:]
    return v + size * 0.5

def vec2skew(v):
  zero = torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype)
  return torch.stack([
    torch.cat([zero, -v[..., 2:3], v[..., 1:2]], dim=-1),
    torch.cat([v[..., 2:3], zero, -v[..., 0:1]], dim=-1),
    torch.cat([-v[..., 1:2], v[..., 0:1], zero], dim=-1),
  ], dim=-2)

def exp(r):
  skew_r = vec2skew(r)
  norm_r = torch.linalg.norm(r).clamp(min=1e-6)
  eye = torch.eye(3, dtype=r.dtype, device=r.device)
  R = eye + \
    (norm_r.sin()/norm_r) * skew_r + \
    ((1 - norm_r.cos())/norm_r.square()) * (skew_r @ skew_r)
  return R

class OrthogonalCamera(Camera):
  def __init__(
    self,
    pos: torch.tensor,
    at: torch.tensor,
    up: torch.tensor,
    view_width:float=10.,
    view_height:float=None,
  ):
    super().__init__()
    if view_height is None: view_height = view_width
    self.pos = pos
    self.dir = dir = F.normalize(at - pos, dim=-1)
    self.right = r = view_width * F.normalize(torch.cross(dir, up), dim=-1)
    self.up = view_height * F.normalize(torch.cross(r, dir), dim=-1)
  def __len__(self): return self.pos.shape[0]
  def __getitem__(self, v):
    # FIXME to make this not use an odd part of the language? it should be fine
    cam = OrthogonalCamera.__new__()
    cam.pos = self.pos[v]
    cam.dir = self.dir[v]
    cam.right = self.up[v]
    cam.up = self.up[v]
    return cam
  def sample_positions(self, position_samples, size:int, with_noise=False):
    u, v = position_samples.split([1,1], dim=-1)
    u = 2 * ((u.squeeze(-1)/size) - 0.5)
    v = 2 * ((v.squeeze(-1)/size) - 0.5)
    r_o = self.pos[:, None, None, :] + \
      u[None, :, :, None] * self.right[:, None, None, :] +\
      v[None, :, :, None] * self.up[:, None, None, :]
    r_d = self.dir.expand_as(r_o)
    return torch.cat([r_o, r_d], dim=-1)

# StaticCamera sits at the origin and has no rotations.
# Should be used when predicting videos with no ground truth camera positions,
# And also allows for the focal parameters to be optimized over
class StaticCamera(Camera):
  def __init__(
    self,
    focal: float=300,
  ):
    super().__init__()
    self.focal = focal
  def __len__(self): return 1
  def __getitem__(self, _v): return self
  def sample_positions(
    self,
    position_samples,
    size:int=512,
    with_noise: bool = False,
  ):
    u,v = position_samples.split([1,1], dim=-1)
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    r_d = torch.stack([
        (u - size * 0.5) / self.focal,
        -(v - size * 0.5) / self.focal,
        -torch.ones_like(u),
    ], dim=-1)
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3)
    r_o = torch.zeros_like(r_d)
    return torch.cat([torch.zeros_like(r_o), r_d], dim=-1)

def lift(x,y,z,intrinsics, size):
    total_shape = x.shape
    fx = intrinsics[..., 0, 0, None].expand(total_shape)
    fy = intrinsics[..., 1, 1, None].expand(total_shape)
    cx = intrinsics[..., 0, 2, None].expand(total_shape) # size of image
    cy = intrinsics[..., 1, 2, None].expand(total_shape) # size of image
    sk = intrinsics[..., 0, 1, None].expand(total_shape)
    x = x.expand(total_shape)
    y = y.expand(total_shape)
    z = z.expand(total_shape)

    x_lift = (x - cx + cy*sk/fy - sk*y/fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack([x_lift, y_lift, z, torch.ones_like(z)], dim=-1)

# A camera specifically for rendering DTU scenes as described in IDR.
class DTUCamera(Camera):
  def __init__(
    self,
    pose: torch.Tensor = None,
    intrinsic: torch.Tensor = None,
  ):
    super().__init__()
    self.pose = nn.Parameter(pose, requires_grad=False)
    self.intrinsic = nn.Parameter(intrinsic, requires_grad=False)

  def __len__(self): return self.pose.shape[0]
  def __getitem__(self, v): return DTUCamera(pose=self.pose[v], intrinsic=self.intrinsic[v])
  def sample_positions(
    self,
    position_samples,
    size:int=512,
    with_noise:bool=False,
  ):
    device = position_samples.device
    pose = self.pose
    intrinsic = self.intrinsic
    # copied directly from https://github.com/lioryariv/idr/blob/main/code/utils/rend_util.py#L48
    if pose.shape[1] == 7: #In case of quaternion vector representation
      raise NotImplementedError()
    # In case of pose matrix representation
    else: r_o = pose[:, :3, 3]

    W, H, _ = position_samples.shape
    N = len(self)

    # 1600, 1200 is magical because it's the size of the original images given to us
    # In theory it would need to be changed if training on a different image set.
    normalize= torch.tensor([1600, 1200], device=device, dtype=torch.float)/size
    u,v = (position_samples * normalize)\
      .reshape(-1, 2)\
      .split([1,1], dim=-1)
    u = u.reshape(1, -1).expand(N, -1)
    v = v.reshape(1, -1).expand(N, -1)

    points = lift(u, v, torch.ones_like(u), intrinsics=intrinsic, size=size)

    world_coords = torch.bmm(pose, points.permute(0,2,1)).permute(0,2,1)[..., :3]

    r_o = r_o[:, None, :].expand_as(world_coords)
    r_d = F.normalize(world_coords - r_o, dim=-1)

    return torch.cat([r_o, r_d], dim=-1).reshape(N, W, H, 6)

