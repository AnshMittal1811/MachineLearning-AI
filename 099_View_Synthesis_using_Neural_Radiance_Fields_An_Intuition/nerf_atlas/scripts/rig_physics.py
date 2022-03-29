import sys
sys.path[0] = sys.path[0][:-len("scripts/")] # hacky way to treat it as root directory

import torch
import argparse
from tqdm import trange

import src.loaders as loaders
from src.utils import ( save_image )
from src.nerf import ( RigNeRF )
from src.physics import ( PointSpringSystem )
import random

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("--model", required=True, type=str, help="Model with points to project")
  a.add_argument("-d", "--data", required=True, type=str, help="Paht to data")
  a.add_argument("--size", type=int, default=128, help="Size to render image at")
  a.add_argument("--data-kind",type=str,default="original")
  a.add_argument("--derive-kind", action="store_true")
  args = a.parse_args()
  setattr(args, "volsdf_alternate", False)
  setattr(args, "bg", "black")
  return args

device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def render(
  model, cam, crop,
  # how big should the image be
  size, args, times=None, with_noise=0.1,
):
  ii, jj = torch.meshgrid(
    torch.arange(size, device=device, dtype=torch.float),
    torch.arange(size, device=device, dtype=torch.float),
    indexing="ij",
  )

  positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
  t,l,h,w = crop
  positions = positions[t:t+h,l:l+w,:]

  rays = cam.sample_positions(positions, size=size, with_noise=with_noise)

  return model(rays), rays

def main():
  args = arguments()
  sz = args.size
  steps = 100
  with torch.no_grad():
    model = torch.load(args.model)
    prev_pts = model.points
    sim = PointSpringSystem(prev_pts)
    # Pick random point, apply huge force in random direction?
    pt_idx = random.randint(0, prev_pts.shape[0]-1)
    F = torch.zeros_like(prev_pts)
    F[pt_idx] = 10*torch.randn(3)
    next_pts = prev_pts + sim(prev_pts, prev_pts, F)
    no_F = torch.zeros_like(F)
    for _ in tqdm(steps):
      delta_x = sim(prev_pts, next_pts, no_F)
      print(delta_x.norm(dim=-1).max())
      prev_pts = next_pts
      next_pts = next_pts + delta_x
  return

if __name__ == "__main__": main()
