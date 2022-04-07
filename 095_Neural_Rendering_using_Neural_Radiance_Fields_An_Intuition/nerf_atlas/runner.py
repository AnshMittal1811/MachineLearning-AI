# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.
import argparse
import random
import json
import math
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import trange, tqdm
from itertools import chain

import src.loaders as loaders
import src.nerf as nerf
import src.utils as utils
import src.sdf as sdf
import src.refl as refl
import src.lights as lights
import src.cameras as cameras
import src.hyper_config as hyper_config
import src.renderers as renderers
from src.lights import light_kinds
from src.opt import UniformAdam
from src.utils import ( save_image, save_plot, load_image, dir_to_elev_azim, git_hash )
from src.neural_blocks import ( Upsampler, SpatialEncoder, StyleTransfer, FourierEncoder )

import os

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ST="store_true"
  a.add_argument("-d", "--data", help="path to data", required=True)
  a.add_argument(
    "--data-kind", help="Kind of data to load", default="original", choices=list(loaders.kinds)
  )
  a.add_argument(
    "--derive-kind", help="Attempt to derive the kind if a single file is given", action="store_false",
  )

  a.add_argument("--outdir", help="path to output directory", type=str, default="outputs/")
  a.add_argument("--timed-outdir", help="Create new output dir with date+time of run", action=ST)

  # various size arguments
  a.add_argument("--size", help="post-upsampling size of output", type=int, default=32)
  a.add_argument("--render-size", help="pre-upsampling size of output image", type=int, default=16)

  a.add_argument("--epochs", help="number of epochs to train for", type=int, default=30000)
  a.add_argument("--batch-size", help="# views pet training batch", type=int, default=8)
  a.add_argument("--neural-upsample", help="Add neural upsampling", action=ST)
  a.add_argument("--crop-size",help="what size to use while cropping",type=int, default=16)
  a.add_argument("--test-crop-size",help="what size to use while cropping at test time",type=int, default=0)
  a.add_argument("--steps", help="Number of depth steps", type=int, default=64)
  a.add_argument(
    "--mip", help="Use MipNeRF with different sampling", type=str, choices=["cone", "cylinder"],
  )
  a.add_argument(
    "--sigmoid-kind", help="What activation to use with the reflectance model.",
    default="upshifted", choices=list(utils.sigmoid_kinds.keys()),
  )

  a. add_argument(
    "--feature-space", help="The feature space size when neural upsampling.",
    type=int, default=32,
  )
  a.add_argument(
    "--model", help="which shape model to use", type=str,
    choices=list(nerf.model_kinds.keys()) + ["sdf"], default="plain",
  )
  a.add_argument(
    "--dyn-model", help="Which dynamic model to use", type=str,
    choices=list(nerf.dyn_model_kinds.keys()),
  )
  a.add_argument(
    "--bg", help="What background to use for NeRF.", type=str,
    choices=list(nerf.sky_kinds.keys()), default="black",
  )
  # this default for LR seems to work pretty well?
  a.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=5e-4)
  a.add_argument("--seed", help="Random seed to use, -1 is no seed", type=int, default=1337)
  a.add_argument("--decay", help="Weight decay value", type=float, default=0)
  a.add_argument("--notest", help="Do not run test set", action=ST)
  a.add_argument("--data-parallel", help="Use data parallel for the model", action=ST)
  a.add_argument(
    "--omit-bg", action=ST, help="Omit black bg with some probability. Only used for faster training",
  )
  a.add_argument(
    "--train-parts", help="Which parts of the model should be trained",
    choices=["all", "refl", "occ", "path-tf", "camera"], default=["all"], nargs="+",
  )
  a.add_argument(
    "--loss-fns", help="Loss functions to use", nargs="+", type=str, choices=list(loss_map.keys()), default=["l2"],
  )
  a.add_argument(
    "--color-spaces", help="Color spaces to compare on", nargs="+", type=str,
    choices=["rgb", "hsv", "luminance", "xyz"], default=["rgb"],
  )
  a.add_argument(
    "--tone-map", help="Add tone mapping (1/(1+x)) before loss function", action=ST,
  )
  a.add_argument("--bendy", help="[WIP] Allow bendy rays!", action=ST)
  a.add_argument(
    "--gamma-correct-loss", type=float, default=1., help="Gamma correct by x in training",
  )
  a.add_argument(
    "--autogamma-correct-loss", action=ST, help="Automatically infer a weight for gamma correction",
  )
  a.add_argument("--has-multi-light", help="For NeRV, use multi point light dataset", action=ST)
  a.add_argument("--style-img", help="Image to use for style transfer", default=None)
  a.add_argument("--no-sched", help="Do not use a scheduler", action=ST)
  a.add_argument(
    "--sched-min", default=5e-5, type=float, help="Minimum value for the scheduled learning rate.",
  )
  a.add_argument("--serial-idxs", help="Train on images in serial", action=ST)
  # TODO really fix MPIs
  a.add_argument(
    "--replace", nargs="*", choices=["refl", "occ", "bg", "sigmoid", "light", "dyn", "al_occ"],
    default=[], type=str, help="Modules to replace on this run, if any. Take caution for overwriting existing parts.",
  )
  a.add_argument(
    "--all-learned-occ-kind", help="What parameters the Learned Ambient Occlusion should take",
    default="pos", type=str, choices=list(renderers.all_learned_occ_kinds.keys()),
  )

  a.add_argument(
    "--volsdf-direct-to-path", action=ST,
    help="Convert an existing direct volsdf model to a path tracing model",
  )
  a.add_argument(
    "--volsdf-alternate", help="Use alternating volume rendering/SDF training volsdf", action=ST,
  )
  a.add_argument(
    "--shape-to-refl-size", type=int, default=64, help="Size of vector passed from density to reflectance model",
  )
  a.add_argument(
    "--refl-order", default=2, type=int, help="Order for classical Spherical Harmonics & Fourier Basis BSDFs/Reflectance models",
  )
  a.add_argument(
    "--inc-fourier-freqs", action=ST, help="Multiplicatively increase the fourier frequency standard deviation on each run",
  )
  a.add_argument("--rig-points", type=int, default=128, help="Number of rigs points to use in RigNeRF")

  refla = a.add_argument_group("reflectance")
  refla.add_argument(
    "--refl-kind", help="What kind of reflectance model to use", choices=list(refl.refl_kinds.keys()), default="view",
  )
  refla.add_argument(
    "--weighted-subrefl-kinds",
    help="What subreflectances should be used with --refl-kind weighted. \
    They will not take a spacial component, and only rely on view direction, normal, \
    and light direction.",
    choices=[r for r in refl.refl_kinds if r != "weighted"], nargs="+", default=[],
  )
  refla.add_argument(
    "--normal-kind", choices=[None, "elaz", "raw"], default=None,
    help="How to include normals in reflectance model. Not all surface models support normals",
  )
  refla.add_argument(
    "--space-kind", choices=["identity", "surface", "none"], default="identity",
    help="Space to encode texture: surface builds a map from 3D (identity) to 2D",
  )
  refla.add_argument(
    "--alt-train", choices=["analytic", "learned"], default="learned",
    help="Whether to train the analytic or the learned model, set per run.",
  )
  refla.add_argument(
    "--refl-bidirectional", action=ST,
    help="Allow normals to be flipped for the reflectance (just Diffuse for now)",
  )
  refla.add_argument(
    "--view-variance-decay", type=float, default=0, help="Regularize reflectance across view directions",
  )

  rdra = a.add_argument_group("integrator")
  rdra.add_argument(
    "--integrator-kind", choices=[None, "direct", "path"], default=None,
    help="Integrator to use for surface rendering",
  )
  rdra.add_argument(
    "--occ-kind", choices=list(renderers.occ_kinds.keys()), default=None,
    help="Occlusion method for shadows to use in integration.",
  )

  rdra.add_argument("--smooth-occ", default=0, type=float, help="Weight to smooth occlusion by.")
  rdra.add_argument(
    "--decay-all-learned-occ", type=float, default=0,
    help="Weight to decay all learned occ by, attempting to minimize it",
  )
  rdra.add_argument(
    "--all-learned-to-joint", action=ST,
    help="Convert a fully learned occlusion model into one with an additional raycasting check"
  )

  lighta = a.add_argument_group("light")
  lighta.add_argument(
    "--light-kind", choices=list(light_kinds.keys()), default=None,
    help="Kind of light to use while rendering. Dataset indicates light is in dataset",
  )
  lighta.add_argument(
    "--light-intensity", type=int, default=100, help="Intensity of light to use with loaded dataset",
  )
  lighta.add_argument(
    "--point-light-position", type=float, nargs="+", default=[0, 0, -3], help="Position of point light",
  )

  sdfa = a.add_argument_group("sdf")
  sdfa.add_argument("--sdf-eikonal", help="Weight of SDF eikonal loss", type=float, default=0)
  sdfa.add_argument("--surface-eikonal", help="Weight of SDF eikonal loss on surface", type=float, default=0)
  # normal smoothing arguments
  sdfa.add_argument("--smooth-normals", help="Amount to attempt to smooth normals", type=float, default=0)
  sdfa.add_argument("--smooth-surface", help="Amount to attempt to smooth surface normals", type=float, default=0)
  sdfa.add_argument(
    "--smooth-eps", help="size of random uniform perturbation for smooth normals regularization",
    type=float, default=1e-3,
  )
  sdfa.add_argument(
    "--smooth-eps-rng", action=ST, help="Smooth by random amount instead of smoothing by a fixed distance",
  )
  sdfa.add_argument(
    "--smooth-n-ord", nargs="+", default=[2], choices=[1,2], type=int,
    help="Order of vector to use when smoothing normals",
  )
  sdfa.add_argument(
    "--sdf-kind", help="Which SDF model to use", type=str,
    choices=list(sdf.sdf_kinds.keys()), default="mlp",
  )
  sdfa.add_argument("--sphere-init", help="Initialize SDF to a sphere", action=ST)
  sdfa.add_argument(
    "--bound-sphere-rad", type=float, default=-1,
    help="Intersect the learned SDF with a bounding sphere at the origin, < 0 is no sphere",
  )
  sdfa.add_argument(
    "--sdf-isect-kind", choices=["sphere", "secant", "bisect"], default="bisect",
    help="Marching kind to use when computing SDF intersection.",
  )

  sdfa.add_argument("--volsdf-scale-decay", type=float, default=0, help="Decay weight for volsdf scale")
  dnerfa = a.add_argument_group("dnerf")
  dnerfa.add_argument(
    "--spline", type=int, default=0, help="Use spline estimator w/ given number of poitns for dynamic nerf delta prediction",
  )
  dnerfa.add_argument("--time-gamma", help="Apply a gamma based on time", action=ST)
  dnerfa.add_argument("--with-canon", help="Preload a canonical NeRF", type=str, default=None)
  dnerfa.add_argument("--fix-canon", help="Do not train canonical NeRF", action=ST)
  dnerfa.add_argument(
    "--render-over-time", default=-1, type=int,
    help="Fix camera to i, and render over a time frame. < 0 is no camera",
  )
  dnerfa.add_argument(
    "--render-over-time-steps", default=100, type=int, help="How many steps to render over time",
  )
  dnerfa.add_argument(
    "--render-over-time-end-sec", default=1, type=float, help="Second to stop rendering"
  )

  cama = a.add_argument_group("camera parameters")
  cama.add_argument("--near", help="near plane for camera", type=float, default=2)
  cama.add_argument("--far", help="far plane for camera", type=float, default=6)
  cama.add_argument("--cam-save-load", help="Location to save/load camera to", default=None)

  vida = a.add_argument_group("Video parameters")
  vida.add_argument("--start-sec", type=float, default=0, help="Start load time of video")
  vida.add_argument("--end-sec", type=float, default=None, help="End load time of video")
  vida.add_argument("--dyn-diverge-decay", type=float, default=0, help="Decay divergence of movement field")
  vida.add_argument("--ffjord-div-decay", type=float, default=0, help="FFJORD divergence of movement field")
  vida.add_argument(
    "--delta-x-decay", type=float, default=0, help="How much decay for change in position for dyn",
  )
  vida.add_argument(
    "--spline-len-decay", type=float, default=0, help="Weight for length of spline regularization"
  )
  vida.add_argument("--spline-pt0-decay", type=float, default=0, help="Add regularization to first point of spline")
  vida.add_argument(
    "--voxel-random-spline-len-decay", type=float, default=0,
    help="Decay for length, randomly sampling a chunk of the grid instead of visible portions",
  )
  vida.add_argument(
    "--random-spline-len-decay", type=float, default=0,
    help="Decay for length, randomly sampling a bezier spline",
  )
  vida.add_argument(
    "--voxel-tv-sigma",
    type=float, default=0, help="Weight of total variation regularization for densitiy",
  )
  vida.add_argument(
    "--voxel-tv-rgb",
    type=float, default=0, help="Weight of total variation regularization for rgb",
  )
  vida.add_argument(
    "--voxel-tv-bezier", type=float, default=0,
    help="Weight of total variation regularization for bezier control points",
  )
  vida.add_argument(
    "--voxel-tv-rigidity", type=float, default=0,
    help="Weight of total variation regularization for rigidity",
  )
  vida.add_argument(
    "--offset-decay", type=float, default=0,
    help="Weight of total variation regularization for rigidity",
  )
  vida.add_argument(
    "--dyn-refl-latent", type=int, default=0,
    help="Size of latent vector to pass from the delta for reflectance",
  )

  vida.add_argument(
    "--render-bezier-keyframes", action=ST, help="Render bezier control points for reference",
  )

  vida.add_argument(
    "--cluster-movement", type=int, default=0,
    help="attempts to visualize clusters of movement into k groups, 0 is off",
  )
  # Long videos
  vida.add_argument(
    "--long-vid-progressive-train", type=int, default=0,
    help="Divide dataset into <N> chunks based on time, train each segment separately",
  )
  vida.add_argument(
    "--long-vid-chunk-len-sec", type=float, default=3,
    help="For a long video, how long should each chunk be in seconds",
  )
  vida.add_argument(
    "--static-vid-cam-angle-deg", type=float, default=40,
    help="Camera angle FOV in degrees, needed for static camera",
  )

  rprt = a.add_argument_group("reporting parameters")
  rprt.add_argument("--name", help="Display name for convenience in log file", type=str, default="")
  rprt.add_argument("-q", "--quiet", help="Silence tqdm", action=ST)
  rprt.add_argument("--save", help="Where to save the model", type=str, default="models/model.pt")
  rprt.add_argument("--save-load-opt", help="Save opt as well as model", action=ST)

  rprt.add_argument("--log", help="Where to save log of arguments", type=str, default="log.json")
  rprt.add_argument("--save-freq", help="# of epochs between saves", type=int, default=5000)
  rprt.add_argument(
    "--valid-freq", help="how often validation images are generated", type=int, default=500,
  )
  rprt.add_argument("--display-smoothness", action=ST, help="Display smoothness regularization")
  rprt.add_argument("--nosave", help="do not save", action=ST)
  rprt.add_argument("--load", help="model to load from", type=str)
  rprt.add_argument("--loss-window", help="# epochs to smooth loss over", type=int, default=250)
  rprt.add_argument("--notraintest", help="Do not test on training set", action=ST)
  rprt.add_argument(
    "--duration-sec", help="Max number of seconds to run this for, s <= 0 implies None",
    type=float, default=0,
  )
  rprt.add_argument(
    "--param-file", type=str, default=None, help="Path to JSON file to use for hyper-parameters",
  )
  rprt.add_argument("--skip-loss", type=int, default=0, help="Number of epochs to skip reporting loss for")
  rprt.add_argument("--msssim-loss", action=ST, help="Report ms-ssim loss during testing")
  rprt.add_argument("--depth-images", action=ST, help="Whether to render depth images")
  rprt.add_argument("--normals-from-depth", action=ST, help="Render extra normal images from depth")
  rprt.add_argument("--depth-query-normal", action=ST, help="Render extra normal images from depth")
  rprt.add_argument("--plt-cmap-kind", type=str, choices=plt.colormaps(), default="magma")
  rprt.add_argument("--gamma-correct", action=ST, help="Gamma correct final images")
  rprt.add_argument("--render-frame", type=int, default=-1, help="Render 1 frame only, < 0 means none")
  rprt.add_argument("--exp-bg", action=ST, help="Use mask of labels while rendering. For vis only")
  rprt.add_argument("--test-white-bg", action=ST, help="Use white background while testing")
  rprt.add_argument("--flow-map", action=ST, help="Render a flow map for a dynamic nerf scene")
  rprt.add_argument("--rigidity-map", action=ST, help="Render a flow map for a dynamic nerf scene")
  # TODO actually implement below
  rprt.add_argument(
    "--visualize", type=str, nargs="*", default=[],
    choices=["flow", "rigidity", "depth", "normals", "normals-at-depth"],
    help="""\
Extra visualizations that can be rendered:
flow: 3d movement, color visualizes direction, and intensity is how much movement
rigidity: how rigid a given region is, higher-intensity is less rigid
depth: how far something is from the camera, darker is closer
normals: surface normals found by raymarching
normals-at-depth: surface normals queried one time at the termination depth of a ray
"""
  )
  rprt.add_argument(
    "--display-regularization", action=ST,
    help="Display regularization in addition to reconstruction loss",
  )
  rprt.add_argument(
    "--y-scale", choices=["linear", "log", "symlog", "logit"], type=str,
    default="linear", help="Scale kind for y-axis",
  )
  rprt.add_argument("--with-alpha", action=ST, help="Render images with an alpha channel")
  # TODO add ability to show all regularization terms and how they change over time.

  meta = a.add_argument_group("meta runner parameters")
  # TODO when using torch jit has problems saving?
  meta.add_argument("--torchjit", help="Use torch jit for model", action=ST)
  meta.add_argument("--train-imgs", help="# training examples", type=int, default=-1)
  meta.add_argument("--draw-colormap", help="Draw a colormap for each view", action=ST)
  meta.add_argument(
    "--convert-analytic-to-alt", action=ST,
    help="Combine a model with an analytic BRDF with a learned BRDF for alternating optimization",
  )
  meta.add_argument("--clip-gradients", type=float, default=0, help="If > 0, clip gradients")
  meta.add_argument("--versioned-save", action=ST, help="Save with versions")
  meta.add_argument(
    "--higher-end-chance", type=int, default=0,
    help="Increase chance of training on either the start or the end",
  )
  meta.add_argument("--opt-step", type=int, default=1, help="Number of steps take before optimizing")

  ae = a.add_argument_group("auto encoder parameters")
  # TODO are these two items still used?
  ae.add_argument("--latent-l2-weight", help="L2 regularize latent codes", type=float, default=0)
  ae.add_argument("--normalize-latent", help="L2 normalize latent space", action=ST)
  ae.add_argument("--encoding-size",help="Intermediate encoding size for AE",type=int,default=32)

  opt = a.add_argument_group("optimization parameters")
  opt.add_argument("--opt-kind", default="adam", choices=list(opt_kinds.keys()), help="What optimizer to use for training")

  args = a.parse_args()

  # runtime checks
  hyper_config.load(args)
  if args.timed_outdir:
    now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    args.outdir = os.path.join(args.outdir, f"{args.name}{'@' if args.name != '' else ''}{now}")
  if not os.path.exists(args.outdir): os.mkdir(args.outdir)

  if not args.neural_upsample:
    args.render_size = args.size
    args.feature_space = 3

  plt.set_cmap(args.plt_cmap_kind)

  assert(args.valid_freq > 0), "Must pass a valid frequency > 0"
  if (args.test_crop_size <= 0): args.test_crop_size = args.crop_size
  return args

opt_kinds = {
  "adam": optim.Adam,
  "sgd": optim.SGD,
  "adamw": optim.AdamW,
  "rmsprop": optim.RMSprop,

  "uniform_adam": UniformAdam,
}
def load_optim(args, params):
  cons = opt_kinds.get(args.opt_kind, None)
  if cons is None: raise NotImplementedError(f"unknown opt kind {args.opt_kind}")
  hyperparams = {
    "lr": args.learning_rate,
    # eps = 1e-7 was in the original paper.
    "eps": 1e-7,
  }
  if args.opt_kind == "adam": hyperparams["weight_decay"] = args.decay
  if args.opt_kind == "sgd": del hyperparams["eps"]
  return cons(params, **hyperparams)

# Computes the difference of the fft of two images
def fft_loss(x, ref):
  got = torch.fft.rfft2(x, dim=(-3, -2), norm="ortho")
  exp = torch.fft.rfft2(ref, dim=(-3, -2), norm="ortho")
  return (got - exp).abs().mean()

# TODO add LPIPS?
loss_map = {
  "l2": F.mse_loss,
  "l1": F.l1_loss,
  "rmse": lambda x, ref: F.mse_loss(x, ref).clamp(min=1e-10).sqrt(),
  "fft": fft_loss,
  "ssim": utils.ssim_loss,
}

color_fns = {
  "hsv": utils.rgb2hsv,
  "luminance": utils.rgb2luminance,
  "xyz": utils.rgb2xyz,
}

# TODO better automatic device discovery here
device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

# DEBUG
#torch.autograd.set_detect_anomaly(True); print("HAS DEBUG")

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


def save_losses(args, losses):
  outdir = args.outdir
  window = args.loss_window

  window = min(window, len(losses))
  losses = np.convolve(losses, np.ones(window)/window, mode='valid')
  losses = losses[args.skip_loss:]
  plt.plot(range(len(losses)), losses)
  plt.yscale(args.y_scale)
  plt.savefig(os.path.join(outdir, "training_loss.png"), bbox_inches='tight')
  plt.close()

def load_loss_fn(args, model):
  if args.style_img != None:
    return StyleTransfer(load_image(args.style_img, resize=(args.size, args.size)))

  # different losses like l1 or l2
  loss_fns = [loss_map[lfn] for lfn in args.loss_fns]
  assert(len(loss_fns) > 0), "must provide at least 1 loss function"
  if len(loss_fns) == 1: loss_fn = loss_fns[0]
  else:
    def loss_fn(x, ref):
      loss = 0
      for fn in loss_fns: loss = loss + fn(x, ref)
      return loss/len(loss_fns)

  assert(len(args.color_spaces) > 0), "must provide at least 1 color space"
  num_color_spaces = len(args.color_spaces)
  # different colors like rgb, hsv
  if num_color_spaces == 1 and args.color_spaces[0] == "rgb":
    # do nothing since this is the default return value
    ...
  elif num_color_spaces == 1:
    cfn = color_fns[args.color_spaces[0]]
    prev_loss_fn = loss_fn
    loss_fn = lambda x, ref: prev_loss_fn(cfn(x), cfn(ref))
  elif "rgb" in args.color_spaces:
    prev_loss_fn = loss_fn
    cfns = [color_fns[cs] for cs in args.color_spaces if cs != "rgb"]
    def loss_fn(x, ref):
      loss = prev_loss_fn(x, ref)
      for cfn in cfns: loss = loss + prev_loss_fn(cfn(x), cfn(ref))
      return loss/num_color_spaces
  else:
    prev_loss_fn = loss_fn
    cfns = [color_fns[cs] for cs in args.color_spaces]
    def loss_fn(x, ref):
      loss = 0
      for cfn in cfns: loss = loss + prev_loss_fn(cfn(x), cfn(ref))
      return loss/num_color_spaces


  if args.tone_map: loss_fn = utils.tone_map(loss_fn)
  if args.gamma_correct_loss != 1.:
    loss_fn = utils.gamma_correct_loss(loss_fn, args.gamma_correct_loss)

  if args.volsdf_alternate:
    return nerf.alternating_volsdf_loss(model, loss_fn, sdf.masked_loss(loss_fn))
  if args.model == "sdf": loss_fn = sdf.masked_loss(loss_fn)
  # if using a coarse fine model, necessary to perform loss on both coarse and fine components.
  if args.model == "coarse_fine":
    prev_loss_fn = loss_fn
    loss_fn = lambda x, ref: prev_loss_fn(model.coarse, ref) + prev_loss_fn(x, ref)
  return loss_fn

def sqr(x): return x * x

# train the model with a given camera and some labels (imgs or imgs+times)
# light is a per instance light.
def train(model, cam, labels, opt, args, sched=None):
  if args.epochs == 0: return

  loss_fn = load_loss_fn(args, model)

  iters = range(args.epochs) if args.quiet else trange(args.epochs)
  update = lambda kwargs: iters.set_postfix(**kwargs)
  if args.quiet: update = lambda _: None

  times = None
  if type(labels) is tuple:
    times = labels[-1].to(device) # oops maybe pass this down from somewhere?
    labels = labels[0]

  batch_size = min(args.batch_size, labels.shape[0])

  get_crop = lambda: (0,0, args.size, args.size)
  cs = args.crop_size
  if cs != 0:
    get_crop = lambda: (
      random.randint(0, args.render_size-cs), random.randint(0, args.render_size-cs), cs, cs,
    )
  train_choices = range(labels.shape[0])
  if args.higher_end_chance > 0:
    train_choices = list(train_choices)
    train_choices += [0] * args.higher_end_chance
    train_choices += [labels.shape[0]-1] * args.higher_end_chance
  next_idxs = lambda _: random.sample(train_choices, batch_size)
  if args.serial_idxs: next_idxs = lambda i: [i%len(cam)] * batch_size
  #next_idxs = lambda i: [i%10] * batch_size # DEBUG

  losses = []
  start = time.time()
  should_end = lambda: False
  if args.duration_sec > 0: should_end = lambda: time.time() - start > args.duration_sec

  train_percent = 1/args.epochs
  opt.zero_grad()
  for i in iters:
    if should_end():
      print("Training timed out")
      break

    curr_percent = train_percent * i
    # goes from 1/100 -> 1 gradually over epochs
    exp_ratio = (1/100) ** (1-curr_percent)


    idxs = next_idxs(i)

    ts = None if times is None else times[idxs]
    c0,c1,c2,c3 = crop = get_crop()
    ref = labels[idxs][:, c0:c0+c2,c1:c1+c3, :3].to(device)

    if getattr(model.refl, "light", None) is not None:
      model.refl.light.set_idx(torch.tensor(idxs, device=device))

    # omit items which are all darker with some likelihood. This is mainly used when
    # attempting to focus on learning the refl and not the shape.
    if args.omit_bg and (i % args.save_freq) != 0 and (i % args.valid_freq) != 0 and \
      ref.mean() + 0.3 < sqr(random.random()): continue

    out, rays = render(model, cam[idxs], crop, size=args.render_size, times=ts, args=args)
    loss = loss_fn(out, ref)
    assert(loss.isfinite()), f"Got {loss.item()} loss"
    l2_loss = loss.item()
    display = {
      "l2": f"{l2_loss:.04f}",
      "refresh": False,
    }
    if sched is not None: display["lr"] = f"{sched.get_last_lr()[0]:.1e}"

    if args.latent_l2_weight > 0: loss = loss + model.nerf.latent_l2_loss * latent_l2_weight

    pts = None
    get_pts = lambda: 5*(torch.randn(((1<<13) * 5)//4 , 3, device=device))
    # prepare one set of points for either smoothing normals or eikonal.
    if args.sdf_eikonal > 0 or args.smooth_normals > 0:
      # NOTE the number of points just fits in memory, can modify it at will
      pts = get_pts()
      n = model.sdf.normals(pts)

    # E[d sdf(x)/dx] = 1, enforces that the SDF is valid.
    if args.sdf_eikonal > 0: loss = loss + args.sdf_eikonal * utils.eikonal_loss(n)
    # E[div(change in x)] = 0, enforcing the change in motion does not compress space.
    if args.dyn_diverge_decay > 0:
      # TODO maybe this is wrong? Unsure
      loss=loss+args.dyn_diverge_decay*utils.divergence(model.pts, model.dp).mean()
    # approximation of divergence using ffjord algorithm as in NR-NeRF
    if args.ffjord_div_decay:
      div_approx = utils.div_approx(model.pts, model.rigid_dp).abs().square()
      loss = loss + exp_ratio * args.ffjord_div_decay * (model.canonical.alpha.detach() * div_approx).mean()
    if args.view_variance_decay > 0:
      pts = pts if pts is not None else get_pts()
      views = torch.randn(2, *pts.shape, device=device)
      refl = model.refl(pts[None].repeat_interleave(2,dim=0), views)
      loss = loss + args.view_variance_decay * F.mse_loss(refl[0], refl[1])

    if args.volsdf_scale_decay > 0: loss = loss + args.volsdf_scale_decay * model.scale_post_act


    # dn/dx -> 0, hopefully smoothes out the local normals of the surface.
    if args.smooth_normals > 0:
      s_eps = args.smooth_eps
      if s_eps > 0:
        if args.smooth_eps_rng: s_eps = random.random() * s_eps
        # epsilon-perturbation implementation from unisurf
        perturb = F.normalize(torch.randn_like(pts), dim=-1) * s_eps
        delta_n = n - model.sdf.normals(pts + perturb)
      else:
        delta_n = torch.autograd.grad(
          inputs=pts, outputs=F.normalize(n, dim=-1), create_graph=True,
          grad_outputs=torch.ones_like(n),
        )[0]
      smoothness = 0
      for o in args.smooth_n_ord:
        smoothness = smoothness + torch.linalg.norm(delta_n, ord=o, dim=-1).sum()
      if args.display_smoothness: display["n-*"] = smoothness.item()
      loss = loss + args.smooth_normals * smoothness

    # smooth_both occlusion and the normals on the surface
    if args.smooth_surface > 0:
      model_ts = model.nerf.ts[:, None, None, None, None]
      depth_region = nerf.volumetric_integrate(model.nerf.weights, model_ts)[0,...]
      r_o, r_d = rays.split([3,3], dim=-1)
      isect = r_o + r_d * depth_region
      perturb = F.normalize(torch.randn_like(isect), dim=-1) * 1e-3
      surface_normals = model.sdf.normals(isect)
      delta_n = surface_normals - model.sdf.normals(isect + perturb)
      smoothness = 0
      for o in args.smooth_n_ord:
        smoothness = smoothness + torch.linalg.norm(delta_n, ord=o, dim=-1).sum()
      if args.display_smoothness: display["n-s"] = smoothness.item()
      loss = loss + args.smooth_surface * smoothness
      if args.surface_eikonal > 0: loss = loss + args.surface_eikonal * utils.eikonal_loss(surface_normals)
      # smooth occ on the surface
    if args.smooth_occ > 0 and args.smooth_surface > 0:
      noise = torch.randn([*isect.shape[:-1], model.total_latent_size()], device=device)
      elaz = dir_to_elev_azim(torch.randn_like(isect, requires_grad=False))
      isect_elaz = torch.cat([isect, elaz], dim=-1)
      att = model.occ.attenuation(isect_elaz, noise).sigmoid()
      perturb = F.normalize(torch.randn_like(isect_elaz), dim=-1) * 5e-2
      att_shifted = model.occ.attenuation(isect_elaz + perturb, noise)
      loss = loss + args.smooth_surface * (att - att_shifted).abs().mean()

    # smoothing the shadow, randomly over points and directions.
    if args.smooth_occ > 0:
      if pts is None:
        pts = 5*(torch.randn(((1<<13) * 5)//4 , 3, device=device, requires_grad=True))
      elaz = dir_to_elev_azim(torch.randn_like(pts, requires_grad=True))
      pts_elaz = torch.cat([pts, elaz], dim=-1)
      noise = torch.randn(pts.shape[0], model.total_latent_size(),device=device)
      att = model.occ.attenuation(pts_elaz, noise).sigmoid()
      perturb = F.normalize(torch.randn_like(pts_elaz), dim=-1) * 1e-2
      att_shifted = model.occ.attenuation(pts_elaz + perturb, noise)
      loss = loss + args.smooth_occ * (att - att_shifted).abs().mean()

    if args.decay_all_learned_occ > 0:
      loss = loss + args.decay_all_learned_occ * model.occ.all_learned_occ.raw_att.neg().mean()

    if args.delta_x_decay > 0: loss = loss + args.delta_x_decay * model.dp.norm(dim=-1).mean()

    # Apply voxel total variation losses
    if args.voxel_tv_sigma > 0: loss = loss + args.voxel_tv_sigma * nerf.total_variation(model.densities)
    if args.voxel_tv_rgb > 0: loss = loss + args.voxel_tv_rgb * nerf.total_variation(model.rgb)
    if args.voxel_tv_bezier > 0: loss = loss + args.voxel_tv_bezier * nerf.total_variation(model.ctrl_pts_grid)
    if args.voxel_tv_rigidity > 0: loss = loss + args.voxel_tv_rigidity * nerf.total_variation(model.rigidity_grid)
    # apply offset loss as described in NR-NeRF
    if args.offset_decay > 0:
      norm_dp = torch.linalg.vector_norm(model.dp, dim=-1, keepdim=True)\
        .pow(2 - model.rigidity)
      reg = model.canonical.weights.detach()[None,...,None] * (norm_dp + 3e-3 * model.rigidity)
      loss = loss + exp_ratio * args.offset_decay * reg.mean()
    # apply regularization on spline length, to get smallest spline that fits.
    # only apply this to visible points though
    if args.spline_len_decay > 0:
      arc_lens = nerf.arc_len(model.ctrl_pts)
      w = model.canonical.weights.detach().squeeze(1)
      loss = loss + args.spline_len_decay * (w * arc_lens).mean()
    if args.spline_pt0_decay > 0 and hasattr(model, "init_p"):
      loss = loss + args.spline_pt0_decay * torch.linalg.norm(model.init_p, dim=-1).mean()
    # TODO is there any way to unify these two? Maybe provide a method to get a random sample on
    # the class?
    if args.voxel_random_spline_len_decay > 0:
      x,y,z = nerf.random_sample_grid(model.ctrl_pts_grid, samples=16**3)
      data = torch.stack(model.ctrl_pts_grid[x,y,z].split(3, dim=-1), dim=0)\
        [:, :, None, None, None, :]
      loss = loss + args.random_spline_len_decay * nerf.arc_len(data).mean()
    if args.random_spline_len_decay > 0:
      if pts is None: pts = 5*(torch.randn(((1<<13) * 5)//4 , 3, device=device, requires_grad=True))
      pts = torch.stack(model.delta_estim(pts)[..., 1:].split(3,dim=-1), dim=0)\
        [:, :, None, None, None, :]
      loss = loss + args.random_spline_len_decay * nerf.arc_len(pts).mean()

    # apply eikonal loss for rendered DynamicNeRF
    if args.sdf_eikonal > 0 and isinstance(model, nerf.DynamicNeRF):
      t = torch.rand(*pts.shape[:-1], 1, device=device)
      dp = model.time_estim(pts, t)
      n_dyn = model.sdf.normals(pts + dp)
      loss = loss + args.sdf_eikonal * utils.eikonal_loss(n_dyn)


    # --- Finished with applying any sort of regularization

    if args.display_regularization: display["reg"] = f"{loss.item():.03f}"

    update(display)
    losses.append(l2_loss)

    assert(loss.isfinite().item()), "Got NaN loss"
    if args.opt_step != 1: loss = loss / args.opt_step
    loss.backward()
    if args.clip_gradients > 0: nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
    if i % args.opt_step == 0:
      opt.step()
      opt.zero_grad()
    if sched is not None: sched.step()
    if args.inc_fourier_freqs:
      for module in model.modules():
        if not isinstance(module, FourierEncoder): continue
        module.scale_freqs()

    # Save outputs within the cropped region.
    if i % args.valid_freq == 0:
      with torch.no_grad():
        ref0 = ref[0,...,:3]
        items = [ref0, out[0,...,:3].clamp(min=0, max=1)]
        if out.shape[-1] == 4:
          items.append(ref[0,...,-1,None].expand_as(ref0))
          items.append(out[0,...,-1,None].expand_as(ref0).sigmoid())

        if args.depth_images and hasattr(model, "nerf"):
          raw_depth = nerf.volumetric_integrate(model.nerf.weights, model.nerf.ts[:, None, None, None, None])
          depth = (raw_depth[0]-args.near)/(args.far - args.near)
          items.append(depth.clamp(min=0, max=1))
          if args.normals_from_depth:
            depth_normal = (50*utils.depth_to_normals(depth)+1)/2
            items.append(depth_normal.clamp(min=0, max=1))
        if args.flow_map and hasattr(model, "rigid_dp"):
          flow_map = nerf.volumetric_integrate(model.nerf.weights, model.rigid_dp)[0]
          flow_map /= flow_map.norm(dim=-1).max()
          flow_map = flow_map.abs().sqrt().copysign(flow_map)
          items.append(flow_map.add(1).div(2))
        if args.rigidity_map and hasattr(model, "rigidity"):
          rigidity_map = nerf.volumetric_integrate(model.nerf.weights, model.rigidity)[0]
          items.append(rigidity_map)
        save_plot(os.path.join(args.outdir, f"valid_{i:05}.png"), *items)

    if i % args.save_freq == 0 and i != 0:
      version = (i // args.save_freq) if args.versioned_save else None
      save(model, cam, args, opt, version)
      save_losses(args, losses)
  # final save does not have a version and will write to original file
  save(model, cam, args, opt)
  save_losses(args, losses)

def test(model, cam, labels, args, training: bool = True):
  times = None
  if type(labels) == tuple:
    times = labels[-1]
    labels = labels[0]

  ls = []
  gots = []
  loss_strings = []

  def render_test_set(model, cam, labels, offset=0):
    with torch.no_grad():
      for i in range(labels.shape[0]):
        ts = None if times is None else times[i:i+1, ...]
        exp = labels[i,...,:3]
        got = torch.zeros_like(exp)
        normals = torch.zeros_like(got)
        depth = torch.zeros(*got.shape[:-1], 1, device=device, dtype=torch.float)
        # RigNeRF visualization
        if isinstance(model, nerf.RigNeRF): proximity_map = torch.zeros_like(exp)
        # dynamic nerf visualization tools
        flow_map = torch.zeros_like(normals)
        rigidity_map = torch.zeros_like(depth)

        if getattr(model.refl, "light", None) is not None:
          model.refl.light.set_idx(torch.tensor([i], device=device))

        if args.test_crop_size <= 0: args.test_crop_size = args.render_size

        cs = args.test_crop_size
        N = math.ceil(args.render_size/cs)
        for x in range(N):
          c0 = x * cs
          for y in range(N):
            c1 = y * cs
            out, rays = render(
              model, cam[i:i+1, ...], (c0,c1,cs,cs), size=args.render_size,
              with_noise=False, times=ts, args=args,
            )
            out = out.squeeze(0)
            got[c0:c0+cs, c1:c1+cs, :] = out

            if hasattr(model, "nerf") and args.depth_images:
              model_ts = model.nerf.ts[:, None, None, None, None]
              depth[c0:c0+cs, c1:c1+cs, :] = \
                nerf.volumetric_integrate(model.nerf.weights, model_ts)[0,...]
            if hasattr(model, "n") and hasattr(model, "nerf") :
              if args.depth_query_normal and args.depth_images:
                r_o, r_d = rays.squeeze(0).split([3,3], dim=-1)
                depth_region = depth[c0:c0+cs, c1:c1+cs]
                isect = r_o + r_d * depth_region
                normals[c0:c0+cs, c1:c1+cs] = (F.normalize(model.sdf.normals(isect), dim=-1)+1)/2
                too_far_mask = depth_region > (args.far - 1e-1)
                normals[c0:c0+cs, c1:c1+cs][too_far_mask[...,0]] = 0
              else:
                render_n = nerf.volumetric_integrate(model.nerf.weights, model.n)
                normals[c0:c0+cs, c1:c1+cs, :] = (render_n[0]+1)/2
            elif hasattr(model, "n") and hasattr(model, "sdf"):
              ...
            if args.flow_map and hasattr(model, "rigid_dp"):
              flow_map[c0:c0+cs,c1:c1+cs] = \
                nerf.volumetric_integrate(model.nerf.weights, model.rigid_dp)
            if args.rigidity_map and hasattr(model, "rigidity"):
              rigidity_map[c0:c0+cs,c1:c1+cs] = \
                nerf.volumetric_integrate(model.nerf.weights, model.rigidity)
            if hasattr(model, "displace"):
              proximity_map[c0:c0+cs,c1:c1+cs] = nerf.volumetric_integrate(
                model.nerf.weights, model.displace.max(dim=-2)[0],
              ).clamp(min=0, max=1)

        gots.append(got)
        loss = F.mse_loss(got, exp)
        psnr = utils.mse2psnr(loss).item()
        ts = "" if ts is None else f",t={ts.item():.02f}"
        o = i + offset
        loss_string = f"[{o:03}{ts}]: L2 {loss.item():.03f} PSNR {psnr:.03f}"
        print(loss_string)
        loss_strings.append(loss_string)
        name = f"train_{o:03}.png" if training else f"test_{o:03}.png"
        if args.gamma_correct:
          exp = exp.clamp(min=1e-10)**(1/2.2)
          got = got.clamp(min=1e-10)**(1/2.2)
        items = [exp, got.clamp(min=0, max=1)]

        if hasattr(model, "n") and hasattr(model, "nerf"): items.append(normals.clamp(min=0, max=1))
        if (depth != 0).any() and args.normals_from_depth:
          depth_normals = (utils.depth_to_normals(depth * 100)+1)/2
          items.append(depth_normals)
        if hasattr(model, "nerf") and args.depth_images:
          depth = (depth-args.near)/(args.far - args.near)
          items.append(depth.clamp(min=0, max=1))
        if args.flow_map and hasattr(model, "rigid_dp"):
          flow_map /= flow_map.norm(keepdim=True, dim=-1).max()
          flow_map = flow_map.abs().sqrt().copysign(flow_map)
          items.append(flow_map.add(1).div(2))
        if args.rigidity_map and hasattr(model, "rigidity"): items.append(rigidity_map)
        if hasattr(model, "displace"): items.append(proximity_map)
        if args.draw_colormap:
          colormap = utils.color_map(cam[i:i+1])
          items.append(colormap)
        if args.exp_bg:
          new_items = []
          for item in items:
            if item.shape[:-1] != labels.shape[1:-1]: new_items.append(item)
            elif item.shape[-1] == 1: new_items.append(item * labels[i,...,3:])
            else: new_items.append(torch.cat([item, labels[i,...,3:]], dim=-1))
          items = new_items
        save_plot(os.path.join(args.outdir, name), *items)
        ls.append(psnr)

  rf = args.render_frame
  if args.render_frame >= 0:
    if hasattr(model.refl, "light"): model.refl.light.set_idx(rf)
    return render_test_set(model, cam[rf:rf+1], labels[rf:rf+1], offset=rf)
  render_test_set(model, cam, labels)
  # also render the multi point light dataset, have to load it separately because it's a
  # slightly different light formulation.
  if args.data_kind == "nerv_point" and args.has_multi_light:
    multi_labels, multi_cams, multi_lights = loaders.nerv_point(
      args.data, training=False, size=args.size,
      light_intensity=args.light_intensity,
      with_mask=False, multi_point=True, device=device,
    )
    model.refl.lights = multi_lights
    render_test_set(model, multi_cams, multi_labels, offset=100)
    labels =  torch.cat([labels, multi_labels], dim=0)

  summary_string = f"""[Summary {args.name} ({"training" if training else "test"}) @ {git_hash()}]:
\tmean {np.mean(ls):.03f}
\tmedian {np.median(ls):.03f}
\tmin {min(ls):.03f}
\tmax {max(ls):.03f}
\tvar {np.var(ls):.03f}"""
  if args.msssim_loss:
    try:
      with torch.no_grad():
        msssim = utils.msssim_loss(gots, labels)
        summary_string += f"\nms-ssim {msssim:.03f}"
    except Exception as e: print(f"msssim failed: {e}")
  print(summary_string)
  with open(os.path.join(args.outdir, "results.txt"), 'w') as f:
    f.write(summary_string)
    for ls in loss_strings:
      f.write("\n")
      f.write(ls)

def render_over_time(args, model, cam):
  cam = cam[args.render_over_time:args.render_over_time+1]
  ts = torch.linspace(0, args.render_over_time_end_sec, steps=args.render_over_time_steps, device=device)
  cs = args.test_crop_size
  N = math.ceil(args.render_size/cs)

  with torch.no_grad():
    for i, t in enumerate(tqdm(ts)):
      got = torch.zeros(args.render_size, args.render_size, 3+args.with_alpha, device=device)
      for x in range(N):
        for y in range(N):
          c0 = x * cs
          c1 = y * cs
          out, _rays = render(
            model, cam, (c0,c1,cs,cs), size=args.render_size,
            with_noise=False, times=t.unsqueeze(0), args=args,
          )
          got[c0:c0+cs, c1:c1+cs, :3] = out.squeeze(0)
          if args.with_alpha: got[c0:c0+cs, c1:c1+cs,3] = model.nerf.weights[:-1].sum(dim=0)
      save_image(os.path.join(args.outdir, f"time_{i:03}.png"), got)

    if not args.render_bezier_keyframes: return

    keyframes = [torch.zeros_like(got) for _ in range(model.spline_n)]
    def render_keyframes(model, cam, crop, size):
      r = torch.arange(size, device=device, dtype=torch.float)
      ii, jj = torch.meshgrid(r,r,indexing="ij")
      positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
      t,l,h,w = crop
      positions = positions[t:t+h,l:l+w,:]

      rays = cam.sample_positions(positions, size=size, with_noise=False)
      return model.render_keyframes(rays)
    for x in range(N):
      for y in range(N):
        c0 = x * cs
        c1 = y * cs
        ks = render_keyframes(model, cam, (c0,c1,cs,cs), size=args.render_size)
        for k, out in zip(ks, keyframes):
          out[c0:c0+cs,c1:c1+cs,:] = k
    for i, k in enumerate(keyframes):
      save_image(os.path.join(args.outdir, f"keyframe_{i:02}.png"), k)

# Sets these parameters on the model on each run, regardless if loaded from previous state.
def set_per_run(model, args, labels):
  if args.epochs == 0: return

  if type(labels) is tuple:
    times = labels[-1]
    labels = labels[0]
  model.nerf.steps = args.steps
  model.nerf.t_near = args.near
  model.nerf.t_far = args.far
  if not isinstance(model, nerf.VolSDF): args.volsdf_scale_decay = 0

  ls = model.intermediate_size # How many extra values the density model outputs

  if "occ" in args.replace:
    assert((args.occ_kind is not None) and hasattr(model, "occ"))
    model.occ = renderers.load_occlusion_kind(args, args.occ_kind, ls).to(device)

  if "al_occ" in args.replace:
    assert(hasattr(model, "occ"))
    replacement = renderers.AllLearnedOcc(ls, kind=args.all_learned_occ_kind).to(device)
    if isinstance(model.occ, renderers.AllLearnedOcc): model.occ = replacement
    elif isinstance(model.occ, renderers.JointLearnedConstOcc): model.occ.alo = replacement
    else: raise NotImplementedError("Does not have AllLearnedOcc to replace")

  if "refl" in args.replace:
    if args.refl_kind != "curr" and hasattr(model, "refl"):
      refl_inst = refl.load(args, args.refl_kind, args.space_kind, ls).to(device)
      model.set_refl(refl_inst)

  if "bg" in args.replace: model.set_bg(args.args)

  if "sigmoid" in args.replace and hasattr(model, "nerf"): model.nerf.set_sigmoid(args.sigmoid_kind)


  if "light" in args.replace:
    if isinstance(model.refl, refl.LightAndRefl):
      model.refl.light = lights.load(args).expand(args.num_labels).to(device)
    else: raise NotImplementedError("TODO convert to light and reflectance")

  if "dyn" in args.replace:
    if isinstance(model, nerf.DynamicNeRF):
      model.set_spline_estim(args.spline) if args.spline > 0 else model.set_delta_estim()
      model = model.to(device)
    else: print("[warn]: Model is not an instance of dynamic nerf, ignoring `--replace dyn.`")

  # converts from a volsdf with direct integration to one with indirect lighting
  if args.volsdf_direct_to_path:
    print("[note]: Converting VolSDF direct integration to path")
    assert(isinstance(model, nerf.VolSDF)), "--volsdf-direct-to-path only applies to VolSDF"
    if model.convert_to_path(): model = model.to(device)
    else: print("[note]: Model already uses pathtracing, nothing changed.")

  if args.all_learned_to_joint:
    assert(hasattr(model, "occ")), "Model must have occlusion parameter for converstion to join"
    if isinstance(model.occ, renderers.JointLearnedConstOcc):
      print("[note]: model already joint learned const, nothing changed.")
    else:
      assert(isinstance(model.occ, renderers.AllLearnedOcc)), "Model occ type must be AllLearnedOcc"
      print("[note]: converting occlusion to Joint Learned Const")
      model.occ = renderers.JointLearnedConstOcc(latent_size=ls,alo=model.occ).to(device)

  if not hasattr(model, "occ") or not isinstance(model.occ, renderers.AllLearnedOcc):
    if args.smooth_occ != 0:
      print("[warn]: Zeroing smooth occ since it does not apply")
      args.smooth_occ = 0
  if args.decay_all_learned_occ > 0:
    if not hasattr(model, "occ"):
      print("[warn]: model does not have occlusion, cannot decay all learned occ")
      args.decay_all_learned_occ = 0
    elif not (isinstance(model.occ, renderers.AllLearnedOcc) or \
      isinstance(model.occ, renderers.JointLearnedConstOcc)):
      print("[warn]: model occlusion is not all-learned, cannot decay all learned occ")
      args.decay_all_learned_occ = 0

  if args.convert_analytic_to_alt:
    assert(hasattr(model, "refl")), "Model does not have a reflectance in the right place"
    if not isinstance(model.refl, refl.AlternatingOptimization) \
      and not (isinstance(model.refl, refl.LightAndRefl) and \
        isinstance(model.refl.refl, refl.AlternatingOptimization)):
      new_alt_opt = lambda old: refl.AlternatingOptimization(
        old_analytic=model.refl.refl,
        latent_size=ls,
        act = args.sigmoid_kind,
        out_features=args.feature_space,
        normal = args.normal_kind,
        space = args.space_kind,
      )
      # need to change the nested feature
      if isinstance(model.refl, refl.LightAndRefl): model.refl.refl = new_alt_opt(model.refl.refl)
      else: model.refl = new_alt_opt(model.refl)
      model.refl = model.refl.to(device)
    else: print("[note]: redundant alternating optimization, ignoring")

  is_voxel = isinstance(model, nerf.NeRFVoxel)
  is_dyn_voxel = isinstance(model, nerf.DynamicNeRFVoxel)
  if is_dyn_voxel: ...
  elif is_voxel and (args.voxel_tv_bezier > 0 or args.voxel_tv_rigidity > 0):
    print("[warn]: model does not have bezier control points or rigidity for static scene")
    args.voxel_tv_bezier = 0
    args.voxel_tv_rigidity = 0
  elif is_voxel: ...
  elif args.voxel_tv_sigma > 0 or args.voxel_tv_rgb > 0 or args.voxel_tv_bezier > 0:
    print("[warn]: model is not voxel, unsetting total variation")
    args.voxel_tv_sigma = 0
    args.voxel_tv_rgb = 0
    args.voxel_tv_bezier = 0
    args.voxel_tv_rigidity = 0

  # swap which portion is being trained for the alternating optimization
  if hasattr(model, "refl"):
    if isinstance(model.refl, refl.AlternatingOptimization): model.refl.toggle(args.alt_train == "analytic")
    elif isinstance(model.refl, refl.LightAndRefl) and isinstance(model.refl.refl, refl.AlternatingOptimization):
      model.refl.refl.toggle(args.alt_train == "analytic")
  if hasattr(model, "refl") and isinstance(model.refl, refl.Positional):
    if args.view_variance_decay > 0:
      print("[warn]: view variance decay unset, positional refl does not use view")
    args.view_variance_decay = 0
  if args.autogamma_correct_loss:
    # based on https://github.com/darrynten/imagick-scripts/blob/master/bin/autogamma
    midrange = 0.5
    weight = (math.log(midrange)/labels.mean().log()).clamp(min=0).item()
    # aggresively adjust to 0.5, to specialize sqrt call in gamma exponent.
    # Also too much gamma correction seems to be prohibitive.
    if weight < 0.55: weight = 0.5
    if weight >= 0.9:
      print(f"[note]: autogamma correct would darken/hardly change image ({weight}), ignoring.")
    else:
      args.gamma_correct_loss = weight
      print(f"[note]: autogamma correction weight is: {weight}")



def load_model(args, light, is_dyn=False):
  if args.model == "sdf": return sdf.load(args, with_integrator=True).to(device)
  model = nerf.load_nerf(args)
  # have to load dyn model before loading refl since it may pass
  # parameters to refl directly.
  if is_dyn: model = nerf.load_dyn(args, model, device)

  # set reflectance kind for new models (but volsdf handles it differently)
  ls = model.intermediate_size
  model.set_refl(refl.load(args, args.refl_kind, args.space_kind, ls))

  if args.data_kind == "pixel-single":
    # args.img is populated in load (single_image)
    model = nerf.SinglePixelNeRF(model, encoder=SpatialEncoder(), img=args.img, device=device)

  if (args.light_kind is not None) and (args.light_kind != "dataset") and (light is None):
    light = lights.load(args).expand(args.num_labels)
    model.refl.light = light

  og_model = model
  # tack on neural upsampling if specified
  if args.neural_upsample:
    upsampler =  Upsampler(
      in_size=args.render_size,
      out=args.size,

      in_features=args.feature_space,
      out_features=3,
    ).to(device)
    # stick a neural upsampling block afterwards
    model = nn.Sequential(model, upsampler, nn.Sigmoid())
    #setattr(model, "nerf", og_model) # TODO how to specify this?

  if args.data_parallel:
    model = nn.DataParallel(model)
    setattr(model, "nerf", og_model)

  if args.volsdf_alternate: model = nerf.AlternatingVolSDF(model)
  if args.torchjit: model = torch.jit.script(model)
  return model.to(device)

def save(model, cam, args, opt, version=None):
  if args.nosave: return
  save = args.save if version is None else f"{args.save}_{version}.pt"
  print(f"Saved to {save}")
  if args.save_load_opt: setattr(model, "opt", opt)
  if args.torchjit: raise NotImplementedError()
  else: torch.save(model, save)

  if args.log is not None:
    setattr(args, "curr_time", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    with open(os.path.join(args.outdir, args.log), 'w') as f:
      json.dump(args.__dict__, f, indent=2)
  if args.cam_save_load is not None: torch.save(cam, args.cam_save_load)

def seed(s):
  if s == -1: return
  torch.manual_seed(s)
  random.seed(s)
  np.random.seed(s)

# entry point into the system
def main():
  args = arguments()
  seed(args.seed)

  labels, cam, light = loaders.load(args, training=True)
  cam = cam.to(device)
  if light is not None: light = light.to(device)

  setattr(args, "num_labels", len(labels))

  is_dyn = type(labels) == tuple

  model = None
  if args.load is not None:
    try: model = torch.load(args.load, map_location=device)
    except Exception as e: print(f"[warn]: Could not load model starting from scratch: {e}")
  if model is None: model = model = load_model(args, light, is_dyn)
  if args.cam_save_load is not None:
    try: cam = torch.load(args.cam_save_load, map_location=device)
    except Exception as e: print(f"[warn]: Failed to load camera: {e}")

  if args.train_imgs > 0:
    if is_dyn: labels = tuple(l[:args.train_imgs, ...] for l in labels)
    else: labels = labels[:args.train_imgs, ...]
    cam = cam[:args.train_imgs, ...]

  set_per_run(model, args, labels)
  light = light if light is not None else getattr(model.refl, "light", None)

  # TODO move this method to another function
  if "all" in args.train_parts: parameters = model.parameters()
  else:
    parameters = []
    if "refl" in args.train_parts:
      assert(hasattr(model, "refl")), "Model must have a reflectance parameter to optimize over"
      parameters.append(model.refl.parameters())
    if "occ" in args.train_parts:
      assert(hasattr(model, "occ")), "Model must have occlusion field (maybe internal bug)"
      parameters.append(model.occ.parameters())
    if "path-tf" in args.train_parts:
      assert(hasattr(model, "transfer_fn")), "Model must have a transfer function"
      parameters.append(model.transfer_fn.parameters())
    parameters = chain(*parameters)
  if "camera" in args.train_parts:
    parameters = chain(parameters, cam.parameters())

  # for some reason AdamW doesn't seem to work for this? Not sure why
  opt = load_optim(args, parameters)
  if args.save_load_opt:
    saved_opt = getattr(model, "opt", None)
    # Do not log if does not exist, commonly will not.
    if saved_opt is not None: opt = saved_opt

  sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.sched_min)
  if args.no_sched: sched = None

  print(f"[info]: git commit {git_hash()}")
  if args.long_vid_progressive_train <= 0: train(model, cam, labels, opt, args, sched=sched)
  else:
    assert(is_dyn), "Can only perform progressive long video training on dynamic datasets"
    segments = args.long_vid_progressive_train
    args.long_vid_progressive_train = 0 # indicate that we saw it already
    assert(args.end_sec is not None), "Must pass a specific end time for progressive training"
    step = (args.end_sec - args.start_sec)/segments
    for i, start_sec in enumerate(np.linspace(args.start_sec, args.end_sec-step, segments)):
      # explicitly put dyn model on CPU, only load certain parts of the model each iter.
      #model = model.cpu()
      # but leave canonical on CUDA, since we always need it.
      #model.canonical.to(device)
      args.start_sec = start_sec
      args.end_sec = start_sec + step
      print(f"[info]: starting progressive section {i}")
      labels, _, _ = loaders.load(args, training=True)
      train(model, cam, labels, opt, args, sched=sched)

  if not args.notraintest: test(model, cam, labels, args, training=True)

  test_labels, test_cam, test_light = loaders.load(args, training=False)
  test_cam = test_cam.to(device)
  if light is not None: test_light = test_light.to(device)

  if test_light is not None: model.refl.light = test_light
  if args.test_white_bg: model.set_bg("white")
  model = model.eval()

  if not args.notest: test(model, test_cam, test_labels, args, training=False)
  if args.render_over_time >= 0: render_over_time(args, model, test_cam)

if __name__ == "__main__": main()

