import matplotlib.pyplot as plt
import numpy as np
import torch
import data_processing.datasets.dataio as dataio
import os
import utils.diff_operators as diff_operators
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import skimage.measure
import cv2
import meta_modules
import scipy.io.wavfile as wavfile
import cmapy

from utils.common_utils import make_contour_plot, min_max_summary
import utils.common_utils as common_utils
import sdf_meshing
from modules import batch_decode
from modules_sdf import SDFIBRNet
import utils.error_metrics as error_metrics
from torchmeta.modules.utils import get_subdict


@torch.no_grad()
def _write_summary_sdf_slices(opt, model: SDFIBRNet, model_input, gt, model_output, writer, total_steps, prefix, params=None):
    """
    Axis aligned slices throught the middle of the SDF volume.
    """
    slice_coords_2d = dataio.get_mgrid(512)

    # YZ slice.
    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input = {'coords': yz_slice_coords.to(model.device)[None, ...]}
    yz_model_out = batch_decode(model.decoder_sdf, yz_slice_model_input,
                                batch_size=opt.batch_size, out_feature_slice=slice(0, 1),
                                params=get_subdict(params, 'decoder_sdf'))

    sdf_values = yz_model_out['model_out'][..., :1]
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

    # XZ slice.
    xz_slice_coords = torch.cat((slice_coords_2d[:, :1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:, -1:]), dim=-1)
    xz_slice_model_input = {'coords': xz_slice_coords.to(model.device)[None, ...]}
    xz_model_out = batch_decode(model.decoder_sdf, xz_slice_model_input,
                                batch_size=opt.batch_size, out_feature_slice=slice(0, 1),
                                params=get_subdict(params, 'decoder_sdf'))

    sdf_values = xz_model_out['model_out'][..., :1]
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

    # XY slice.
    xy_slice_coords = torch.cat((slice_coords_2d[:, :2],
                                 torch.zeros_like(slice_coords_2d[:, :1])), dim=-1)
    xy_slice_model_input = {'coords': xy_slice_coords.to(model.device)[None, ...]}
    xy_model_out = batch_decode(model.decoder_sdf, xy_slice_model_input,
                                batch_size=opt.batch_size, out_feature_slice=slice(0, 1),
                                params=get_subdict(params, 'decoder_sdf'))

    sdf_values = xy_model_out['model_out'][..., :1]
    sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

    # min_max_summary(prefix + 'model_out_min_max', model_output['sdf_out'][..., :1], writer, total_steps)
    # min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
