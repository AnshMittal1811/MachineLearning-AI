import sys
sys.path[0] = sys.path[0][:-len("scripts/")] # hacky way to treat it as root directory

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.neural_blocks import ( SkipConnMLP, StyleTransfer, FourierEncoder )
from src.utils import ( load_image, save_image )
from tqdm import trange
import random
import math

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument(
    "--image", type=str, required=True, help="Which image to run this on",
  )
  a.add_argument("--size", type=int, default=180, help="Size of image to render")
  a.add_argument("--epochs", type=int, default=10_000, help="Number of iterations to render for")
  a.add_argument("--latent-size", type=int, default=16, help="Latent size for image model")
  a.add_argument(
    "--batch-size", type=int, default=2,
    help="Number of different times to compute at concurrently",
  )
  a.add_argument(
    "--valid-freq", type=int, default=250, help="How frequently to save output images",
  )
  return a.parse_args()

@torch.jit.script
def wide_sigmoid(x, eps:float=1e-2): return x.sigmoid() * (1 + 2*eps) - eps

class SmoothImageApprox(nn.Module):
  def __init__(
    self,
    latent_size=32,
  ):
    super().__init__()
    self.ls = latent_size
    self.approx = SkipConnMLP(
      in_size=2, out=3, latent_size=latent_size,
      enc=FourierEncoder(input_dims=2, freqs=32),
      num_layers=5, hidden_size=256,
      init="siren", activation=torch.sin,
    )
    self.displacement = SkipConnMLP(
      in_size=3, out=1, latent_size=latent_size,
      hidden_size=256, activation=torch.sin,
    )
  # initializes this approximation at 0 to be the correct image
  def init_zero(self, image, epochs=750):
    sz = image.shape[0]
    device = image.device
    points = torch.stack(torch.meshgrid(
      torch.linspace(-1, 1, steps=sz, device=device),
      torch.linspace(-1, 1, steps=sz, device=device),
    ), dim=-1)
    opt = optim.Adam(self.approx.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    t = trange(epochs)
    for i in t:
      opt.zero_grad()
      Z = torch.randn(sz,sz,self.ls,device=device)
      got = wide_sigmoid(self.approx(points, Z))
      loss = F.mse_loss(got, image).sqrt()
      loss.backward()
      t.set_postfix(l2=f"{loss.item():.03f}")
      opt.step()
      sched.step()
    return got.detach()
  def forward(self, x, t, latent):
    angle = self.displacement(torch.cat([x, t], dim=-1), latent)
    angle = torch.remainder(angle, 2*math.pi)
    dx = torch.cat([angle.cos(), angle.sin()], dim=-1)
    # Force there to be some movement
    dx = F.normalize(dx, dim=-1, eps=1e-6) * (t * 2 * math.pi).sin()
    return wide_sigmoid(self.approx(torch.fmod(x + dx, 1), latent))

device="cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def train(
  model,
  image,
  D,
  opt,

  args,
):
  t = trange(args.epochs)
  B = args.batch_size
  ls = args.latent_size
  sz = args.size
  points = torch.stack(torch.meshgrid(
    torch.linspace(-1, 1, steps=args.size, device=device),
    torch.linspace(-1, 1, steps=args.size, device=device),
  ), dim=-1)[None, ...].expand(B, -1, -1, -1)

  for i in t:
    opt.zero_grad()
    # make multiple times and train them at the same time
    time = torch.randn(B,1,1,1,device=device).expand(B,sz,sz,1)
    latent = torch.randn(ls, device=device)[None, None, :]\
      .expand([*points.shape[:-1],-1])
    got = model(points, time, latent)
    style_loss, content_loss = 0, 0
    for g in got.split(1, dim=0):
      sl, cl = D(g.permute(0, 3, 1, 2))
      style_loss = style_loss + sl
      #content_loss = content_loss + cl
    loss = style_loss + content_loss
    t.set_postfix(style=f"{style_loss:.03f}", content=f"{content_loss:.03f}")
    loss.backward()
    opt.step()
    if i % args.valid_freq == 0:
      save_image(f"outputs/fieldgan_{i:05}.png", got[0].clamp(min=0, max=1))

def test(model, args):
  with torch.no_grad():
    ls = args.latent_size
    points = torch.stack(torch.meshgrid(
      torch.linspace(-1, 1, steps=args.size, device=device),
      torch.linspace(-1, 1, steps=args.size, device=device),
    ), dim=-1)
    latent1 = torch.randn(ls, device=device)[None, None, :]\
      .expand([*points.shape[:-1],-1])
    latent2 = torch.randn(ls, device=device)[None, None, :]\
      .expand([*points.shape[:-1],-1])
    steps = 100
    for i, t in enumerate(torch.linspace(0, 1, steps=steps, device=device)):
      t = t.expand([*points.shape[:-1], 1])
      alpha = i/steps
      latent = (1-alpha) * latent1 + alpha * latent2
      got = model(points, t, latent)
      save_image(f"outputs/fieldgan_test_{i:03}.png", got)

    latent3 = torch.randn(ls, device=device)[None, None, :]\
      .expand([*points.shape[:-1],-1])
    for i, t in enumerate(reversed(torch.linspace(0, 1, steps=steps, device=device))):
      t = t.expand([*points.shape[:-1], 1])
      alpha = i/steps
      latent = (1-alpha) * latent2 + alpha * latent3
      got = model(points, t, latent)
      i = i + steps
      save_image(f"outputs/fieldgan_test_{i:03}.png", got)


def main():
  args = arguments()
  model = SmoothImageApprox(latent_size=args.latent_size).to(device)
  image = load_image(args.image, [args.size, args.size])
  save_image(f"outputs/fieldgan_ref.png", image)
  image = image.permute(2, 0, 1)
  # use same image for content and loss.
  D = StyleTransfer(image[None, ...], image[None, ...]).to(device)
  image = image.to(device)
  init_image = model.init_zero(image.permute(1,2,0))
  save_image(f"outputs/fieldgan_init.png", init_image)
  opt = optim.Adam(model.displacement.parameters(), lr=1e-3, weight_decay=0)
  #opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
  train(model, image, D, opt, args)
  # TODO render the image after, fixing a latent noise and iterating through time
  test(model, args)

if __name__ == "__main__": main()
