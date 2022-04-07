# march.py contains a bunch of SDF marching algorithms.
# all functions should return (pts, hits, dist, None|throughput)
# throughput should be returned if computed, otherwise it should just return None.
# What is throughput? Throughput is the minimum SDF value along a ray. When returned, the SDF
# will be differentiable w.r.t. the throughput.

import torch
import torch.nn.functional as F
import torch.optim as optim
import random

def load_intersection_kind(kind):
  if kind == "sphere": return sphere_march
  if kind == "secant": return secant
  if kind == "bisect": return bisect
  # TODO first intersect with sphere marching using small # of iters,
  # then intersect with bisection/secant. Seems to work for IDR (and PhySG which took from IDR)
  if kind == "march": raise NotImplementedError("")

  raise NotImplementedError(f"unknown intersection kind {kind}")

# sphere_march is a traditional sphere marching algorithm on the SDF.
# It returns the (pts: R^3s, mask: bools, t: step along rays)
#
# note that this implementation is efficient in that it only will compute distance
# for pts that are still candidates.
def sphere_march(
  self,
  r_o, r_d,
  iters: int = 32,
  eps: float = 1e-3,
  near: float = 0, far: float = 1,
):
  device = r_o.device
  with torch.no_grad():
    hits = torch.zeros(r_o.shape[:-1] + (1,), dtype=torch.bool, device=device)
    rem = torch.ones_like(hits).squeeze(-1)
    curr_dist = torch.full_like(hits, near, dtype=torch.float)
    for i in range(iters):
      curr = r_o[rem] + r_d[rem] * curr_dist[rem]
      dist = self(curr)[...,0].reshape_as(curr_dist[rem])
      hits[rem] |= ((dist < eps) & (curr_dist[rem] <= far))
      # anything that was hit or is past range no longer need to compute
      curr_dist[rem] += dist
      rem[hits.squeeze(-1) | (curr_dist > far).squeeze(-1)] = False
    curr = r_o + r_d * curr_dist
  return curr, hits.squeeze(-1), curr_dist, None

# finds an intersection with secant intersection
def secant(
  self,
  r_o, r_d,
  iters: int = 128,
  eps: float = 1e-3,
  near: float = 0, far: float = 1,
):
  tput, best_pos, last_pos, first_neg = throughput_with_sign_change(self, r_o, r_d, near, far, batch_size=iters)
  pts = secant_find(self, r_o, r_d, near=last_pos, far = first_neg, iters=iters)
  hits = tput < 0
  return pts, hits, best_pos, tput

# finds an intersection with secant intersection
def bisect(
  self,
  r_o, r_d,
  iters: int = 128,
  eps: float = 0,
  near: float = 0, far: float = 1,
):
  tput, best_pos, last_pos, first_neg = throughput_with_sign_change(
    self, r_o, r_d, near=near, far=far, batch_size=iters,
  )
  pts = bisection(self, r_o, r_d, near=last_pos, far = first_neg, iters=min(32, iters))
  hits = tput < 0
  return pts, hits, best_pos, tput.unsqueeze(-1)

# computes throughput as well positions where the signs change
def throughput_with_sign_change(
  self,
  r_o, r_d,
  near: float,
  far: float,
  batch_size:int = 128,
):
  # some random jitter I guess?
  max_t = far-near+random.random()*(2/batch_size)
  step = max_t/batch_size
  with torch.no_grad():
    sd = self(r_o + near)[...,0]
    curr_min = sd
    idxs = torch.zeros_like(sd, dtype=torch.long)
    # pos and neg indeces
    last_pos = torch.full_like(sd, -1, dtype=torch.long)
    first_neg = torch.full_like(sd, -1, dtype=torch.long)
    for i in range(batch_size):
      t = near + step * (i+1)
      sd = self(r_o + t * r_d)[..., 0]
      idxs = torch.where(sd < curr_min, i+1, idxs)
      curr_min = torch.minimum(curr_min, sd)
      mask = (first_neg == -1) & (sd < 0)
      last_pos = torch.where(mask, i, last_pos)
      first_neg = torch.where(mask, i + 1, first_neg)
    idxs = idxs.unsqueeze(-1)
    # TODO return best distances.
    # convert from indeces to t
    best_pos = r_o  + (near + idxs * step) * r_d
    first_neg = first_neg.unsqueeze(-1) * step
    last_pos = last_pos.unsqueeze(-1) * step
  val = self(best_pos)
  return val[...,0], best_pos, last_pos, first_neg

# secant marching as implemented in IDR. It seems kind of broken, no idea how it works in their
# implementation.
def secant_find(
  self,
  r_o, r_d,
  near, far,
  iters: int = 32,
):
  device = r_o.device
  with torch.no_grad():
    low = near
    high = far
    sdf_low = self(r_o + low * r_d)[..., 0, None]
    sdf_high = self(r_o + high * r_d)[..., 0, None]
    z_pred = -sdf_low * (high - low) / (sdf_high - sdf_low).clamp(min=1) + low
    assert(z_pred.isfinite().all()), z_pred[~z_pred.isfinite()]
    for i in range(iters):
      mid = r_o + z_pred * r_d
      sdf_mid = self(mid)[..., 0, None]
      ...
      low_mask = sdf_mid > 0
      low[low_mask] = z_pred[low_mask]
      sdf_low[low_mask] = sdf_mid[low_mask]
      ...
      high_mask = sdf_mid < 0
      high[high_mask] = z_pred[high_mask]
      sdf_high[high_mask] = sdf_mid[high_mask]
      ...

      z_pred = -sdf_low * (high - low) / (sdf_high - sdf_low).clamp(min=1) + low
  assert(z_pred.isfinite().all()), z_pred[~z_pred.isfinite()]
  return r_o + z_pred * r_d

# bisection similar to what is implemented in PhySG, which is identical to secant marching but
# bisection by taking the midpoint.
def bisection(
  self,
  r_o, r_d,
  near, far,
  iters: int = 32,
  # different eps than elsewhere, what is considered done
  eps=1e-6,
):
  device = r_o.device
  with torch.no_grad():
    low = near
    high = far
    assert((high >= low).all())
    sdf_low = self(r_o + low * r_d)[..., 0, None]
    sdf_high = self(r_o + high * r_d)[..., 0, None]
    todo = ((high - low) > eps) & (sdf_low > 0) & (sdf_high < 0) & (high > low)
    z_pred = (low + high)/2
    for i in range(iters):
      if not todo.any(): break
      mid = r_o + z_pred * r_d
      sdf_mid = self(mid)[..., 0, None]
      ...
      low_mask = (sdf_mid > 0) & todo
      low[low_mask] = z_pred[low_mask]
      sdf_low[low_mask] = sdf_mid[low_mask]
      ...
      high_mask = (sdf_mid < 0) & todo
      high[high_mask] = z_pred[high_mask]
      sdf_high[high_mask] = sdf_mid[high_mask]
      ...

      z_pred = (low + high)/2
      todo = todo & ((high - low) > eps) & (sdf_low > 0) & (sdf_high < 0) & (high > low)
  return r_o + z_pred * r_d

def throughput(
  self,
  r_o, r_d,
  near: float, far: float,
  batch_size:int = 128,
):
  assert(far > near)
  # some random jitter I guess?
  max_t = far-near+random.random()*(2/batch_size)
  step = max_t/batch_size
  with torch.no_grad():
    sd = self(r_o + near * r_d)[...,0]
    curr_min = sd
    idxs = torch.zeros_like(sd, dtype=torch.long, device=r_d.device)
    for i in range(batch_size):
      t = near + step * (i+1)
      sd = self(r_o + t * r_d)[..., 0]
      idxs = torch.where(sd < curr_min, i+1, idxs)
      curr_min = torch.minimum(curr_min, sd)
    idxs = idxs.unsqueeze(-1)
    best_pos = r_o  + (near + idxs * step) * r_d
  return self(best_pos)[...,0], best_pos
