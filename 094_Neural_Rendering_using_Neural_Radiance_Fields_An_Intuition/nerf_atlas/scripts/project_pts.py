import sys
sys.path[0] = sys.path[0][:-len("scripts/")] # hacky way to treat it as root directory

import torch
import argparse
from tqdm import trange

import src.loaders as loaders
from src.utils import ( save_image )
from src.nerf import ( RigNeRF )

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

  if times is not None: return model((rays, times)), rays
  elif args.data_kind == "pixel-single": return model((rays, positions)), rays
  return model(rays), rays

def main():
  args = arguments()
  sz = args.size
  with torch.no_grad():
    model = torch.load(args.model)
    exit()
    assert(isinstance(model, RigNeRF)), "Can only project pts of RigNeRF"
    labels, cam, _ = loaders.load(args, training=False, device=device)
    for i in trange(labels.shape[0]):
      c = cam[i:i+1]
      pt2d = c.project_pts(model.points, sz)
      out = torch.zeros(sz, sz, 3, dtype=torch.float, device=device)
      pixels = pt2d.long()
      pixels = pixels[((0 <= pixels) & (pixels < sz)).all(dim=-1)]
      out[pixels[:, 0], pixels[:, 1]] = 1
      save_image(f"outputs/proj_pts_{i:03}.png", out)
  return

if __name__ == "__main__": main()
