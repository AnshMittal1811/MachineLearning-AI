"""
Renders previously trained SDF+Color model.
"""
# Enable import from parent package
from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

# Import utils early to initialize matplotlib.
import utils.common_utils as common_utils

import os
from functools import partial
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import imageio
import torch
import numpy as np
import utils.error_metrics as error_metrics

import sdf_meshing
from experiment_scripts.train_sdf_ibr import get_sdf_decoder, get_arg_parser, get_dataset, get_latest_checkpoint_file
from data_processing.components.conversions import imwritef
import modules_sdf
import data_processing.datasets.dataio_sdf as dataio_sdf
from data_processing.datasets.image_view import ImageView
from utils import math_utils


def try_load_optimizer(opt, checkpoint_file):
    """
    Try to load optimizer checkpoint.
    """
    checkpoint_file = Path(checkpoint_file)
    name = common_utils.checkpoint_to_optim_name(checkpoint_file.stem)
    optim_file = checkpoint_file.parent / f'{name}.pth'
    if not optim_file.is_file():
        return None
    return torch.load(optim_file, map_location=opt.device)


def raytrace_view(filename,
                  opt,
                  model: modules_sdf.SDFIBRNet,
                  dataset: dataio_sdf.DatasetSDF,
                  view: ImageView,
                  timestamp: float = common_utils.KEY_FRAME_TIME):

    view_size = (view.resolution * opt.im_scale + 0.5).astype(int)
    projection_matrix = view.projection_matrix
    if opt.image_crop is not None:
        # Optionally crop the viewport.
        opt.image_crop = np.array(opt.image_crop, int)
        crop_gt, projection_matrix = view.crop_image(
            view.image, projection_matrix, opt.image_crop)
        imageio.imwrite(str(filename) + '_crop_gt.png', crop_gt.permute(1, 2, 0).numpy())
        view_size = opt.image_crop[2:]

    res = sdf_meshing.raytrace_sdf_ibr(model,
                                       resolution=view_size,
                                       projection_matrix=projection_matrix,
                                       view_matrix=view.view_matrix,
                                       model_matrix=dataset.model_matrix,
                                       build_pcd=False,
                                       render_softmask=False,
                                       batch_size=opt.batch_size,
                                       debug_gui=False,
                                       timestamp=timestamp)

    if dataset.coords.shape[0] > 0:
        # Compute 3D errror.
        error = error_metrics.compute_pcd_error(dataset.coords, res)

    # Save images.
    sdf_meshing.write_raytrace_results(res, filename, gui=False)


def main():
    p = get_arg_parser()
    p.add_argument('--is_test_only', type=int, default=1, help='Ignore train infra.')
    p.add_argument('--checkpoint_path_test', type=str, default=None, help='Explicit checkpoint path.')
    p.add_argument('--image_crop', type=int, nargs=4, default=None, help="Image crop bounding box for the ray tracing.")
    p.add_argument('--test_plot', type=int, default=1, help='Do Z plot?')
    p.add_argument('--test_rt', type=int, default=1, help='Do raytracing?')
    p.add_argument('--test_spacetime_video', type=int, default=1, help='Raytrace a spacetime video?')
    p.add_argument('--test_cameras', type=int, default=1, help='Render camera poses?')
    p.add_argument('--test_metrics', type=int, default=1, help='Compute metrics?')
    p.add_argument('--video_num_frames', type=int, default=150, help='Number of frames at 30 FPS')
    p.add_argument('--video_yaw', type=float, default=180, help='Video yaw range [deg]')
    p.add_argument('--video_pivot_offset', type=float, nargs=3,
                   default=[0, 0, 0], help='Offset of camera pivot in normalized space [-1,1]')
    p.add_argument('--video_save_frames', type=int, default=0, help='Save frames to images?')
    p.add_argument('--video_path_type', type=str, default='eight', help='Path type [linear|eight]')
    p.add_argument('--benchark_all_views', type=int, default=0, help='Benchmark all views?')
    opt = p.parse_args()

    # Clear the stringified None.
    for k, v in vars(opt).items():
        if p.get_default(k) is None and v == 'None':
            setattr(opt, k, None)
    opt.ibr_dataset = 1
    opt.source_views_per_target = -1

    opt.im_scale = 1
    if opt.dataset_name == 'dtu':
        opt.load_im_scale = 0.5
        WITHHELD_VIEWS = [12, 32, 40]
    elif opt.dataset_name == 'nlr':
        opt.load_im_scale = 0.2
        WITHHELD_VIEWS = []
    elif opt.dataset_name == 'shapenet':
        opt.load_im_scale = 1
        WITHHELD_VIEWS = [7, 16, 23]

    # Prepare output path.
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    # Dataset (needed for metadata and some debug tests)
    dataset = get_dataset(opt)

    im_size = (dataset.resolution * opt.im_scale + 0.5).astype(int)

    # Model
    model = get_sdf_decoder(opt, dataset=dataset)
    model.eval()

    # Ignore init checkpoints specified in the training script.
    if opt.config_filepath and Path(opt.config_filepath).stem == 'args':
        opt.checkpoint_path = None
        opt.experiment_name = Path(opt.config_filepath).parent.stem

    # Load checkpoint.
    if opt.checkpoint_path_test and os.path.isfile(opt.checkpoint_path_test):
        checkpoint_file = Path(opt.checkpoint_path_test)
    elif opt.checkpoint_path and os.path.isfile(opt.checkpoint_path):
        checkpoint_file = Path(opt.checkpoint_path)
    else:
        checkpoint_file = get_latest_checkpoint_file(opt)

    if not checkpoint_file:
        print("WARNING: No checkpoint file found!")
        sink_name = 'empty'
    else:
        print(f'Loading checkpoint from {checkpoint_file}...')
        model.load_state_dict(torch.load(checkpoint_file, map_location=opt.device), strict=False)
        sink_name = checkpoint_file.stem
        # Try to load optimizer.
        opt.optimizer_state = try_load_optimizer(opt, checkpoint_file)

    print(f'Precomputing 3D position buffers.')
    model.precompute_3D_buffers()

    # Prepare output path.
    out_filename = Path(root_path) / 'meshes' / sink_name / 'test'
    common_utils.cond_mkdir(str(out_filename.parent))

    print(f'Reference view = {opt.reference_view} => {dataset.reference_view_index}')

    # Plot SDF.
    if opt.test_plot:
        fig = sdf_meshing.plot_sdf(model)
        plt.savefig(str(out_filename) + '_plot.png')
        plt.close(fig)

    # Render camera poses.
    if opt.test_cameras:
        sdf_meshing.render_cameras(model, dataset, str(out_filename), resolution=512, batch_size=opt.batch_size)

    # Execute the tracer.
    if opt.test_rt:
        raytrace_view(str(out_filename) + f'_view_{dataset.reference_view_index:02d}',
                      opt, model, dataset, dataset.reference_view, common_utils.KEY_FRAME_TIME)

    # Ray trace video of view swipe through time.
    if opt.test_spacetime_video:
        timestamps = np.linspace(-1, 1, dataset.dataset_img.num_frames)

        # This line can be changed to get another list of views for the video. View dataset file for other types of
        # camera paths
        views = dataset.get_camera_path_interp(opt.video_num_frames)
        sdf_meshing.raytrace_video_ibr(str(out_filename) + '_spacetime_video',
                                       model,
                                       resolution=im_size,
                                       projection_matrix=dataset.projection_matrix,
                                       view_matrices=views,
                                       model_matrix=dataset.model_matrix,
                                       timestamps=timestamps,
                                       batch_size=opt.batch_size,
                                       render_diffuse=False,
                                       save_frames=opt.video_save_frames,
                                       debug_gui=False)

    # Compute metrics.
    if opt.test_metrics:
        error_metrics.measure_error(str(out_filename), opt, model, dataset)


if __name__ == "__main__":
    main()
