"""
Trains the IBR model using 3D or 2D multi-view data, and uses meta-learning on the
shape representation model.
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

import torch
from torch.utils.data import DataLoader
import configargparse

import data_processing.datasets.dataio_sdf as dataio_sdf
import data_processing.datasets.dataio_meta as dataio_meta
from utils.ray_builder import RayBuilder
import utils.utils_sdf as utils_sdf
import utils.utils_meta as utils_meta
import training
import loss_functions
import modules_sdf
import meta_modules


def get_arg_parser():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    # Save/resume.
    p.add_argument('--logging_root', type=str, default='./logs_meta_ibr_final', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                   help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help='Checkpoint to trained model. Latest used as default.')
    p.add_argument('--checkpoint_strict', type=int, default=1,
                   help='Is the checkpoint strict (containing all modules)?')
    p.add_argument('--checkpoint_sdf', type=str, default=None,
                   help='Checkpoint to only use for SDF. Overrides defaults.')
    p.add_argument('--checkpoint_img_encoder', type=str, default=None,
                   help='Checkpoint to only use for Image Encoder.')
    p.add_argument('--checkpoint_img_decoder', type=str, default=None,
                   help='Checkpoint to only use for Image Decoder.')
    p.add_argument('--checkpoint_aggregation_mlp', type=str, default=None,
                   help='Checkpoint to only use for Aggregation MLP.')
    p.add_argument('--resume', type=int, default=1,
                   help='Resume from previous checkpoint?')
    p.add_argument('--restart', type=int, default=1,
                   help='Remove all prev checkpoints and summaries in the log dir?')
    p.add_argument('--verbose_logging', type=int, default=0,
                   help='Save complete state every single iteration.')
    p.add_argument('--load_verbose_record', type=str, default=None,
                   help='Loads verbose record for debugging.')
    p.add_argument('--load_model_poses', type=int, default=0,
                   help='Load model poses from the trained SDF network. This must be set when there is a '
                        'dataset mismatch')

    # General training options
    p.add_argument('--device', type=str, default='cuda', help='Device to use.')
    p.add_argument('--batch_size', type=int, default=32768, help='Number of points for 3D supervision')
    p.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for.')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--lr_sdf', type=float, default=5e-5, help='learning rate for sdf. default=5e-5.')
    p.add_argument('--lr_decay_factor', type=float, default=0.5, help='How omuch to decay LR.')
    p.add_argument('--lr_sdf_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_encdec_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_agg_decay_steps', type=int, default=0, help='How often to decay LR.')
    p.add_argument('--lr_alternating_interval', type=int, default=0,
                   help='How often (steps) to swap color and sdf training.')

    p.add_argument('--epochs_til_ckpt', type=int, default=1,
                   help='Time interval in epochs until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=1,
                   help='Time interval in steps until tensorboard summary is saved.')
    p.add_argument('--image_batch_size', type=int, default=4, help='Number of target images per batch.')

    # Implicit Models
    p.add_argument('--model', type=str, default='ours',
                   help='Predefined models [ours|idr|deepsdf]')
    p.add_argument('--model_activation_sdf', type=str, default='sine',
                   help='Activation function [sine|relu]')
    p.add_argument('--model_hidden_layers_sdf', type=int, default=3,
                   help='How many hidden layers between 1st and last.')
    p.add_argument('--model_hidden_dims_sdf', type=int, default=256,
                   help='How many dimensions in hidden layers.')
    p.add_argument('--model_skips_sdf', type=str, default='none',
                   help='Comma separated skip connections.')
    p.add_argument('--feature_vector', type=int, default=-1,
                   help='IDR-like feature vector size.')

    # CNN Models
    p.add_argument('--model_image_encoder_depth', type=int, default=3,
                   help='Depth of the ResNet used for encoding individual images.')
    p.add_argument('--model_image_encoder_features', type=int, default=16,
                   help='Size of output features of each image.')
    p.add_argument('--model_image_decoder_depth', type=int, default=2,
                   help='Depth of the UNet used for decoding an image from feature.')
    p.add_argument('--feature_type', type=str, default='learned',
                   help='Type of features. Whether they are learned, or simply RGB values from input '
                        'views. [learned|rgb]')

    # Feature aggregation methods
    p.add_argument('--feature_aggregation_method', type=str, default='sum',
                   help='Feature aggregation method [sum|lumigraph(_epipolar)|mean|mlp].')
    p.add_argument('--source_views_per_target', type=int, default=25,
                   help='Number of source views to sample features from to render target.')
    p.add_argument('--total_number_source_views', type=int, default=-1,
                   help='Total number of source views to select from.')
    p.add_argument('--source_view_selection_mode', type=str, default='random',
                   help='Method for selecting source views to sample features from [random|nearest]')
    p.add_argument('--occlusion_method', type=str, default='raytrace',
                   help='Method for deciding if features are occluded or not')

    # Positional encoding.
    p.add_argument('--posenc_sdf', type=str, default='none',
                   help='Positional encoding for SDF [none|nerf|idr|ff].')
    p.add_argument('--posenc_sdf_bands', type=int, default=0,
                   help='Number of pos enc bands.')
    p.add_argument('--posenc_sdf_sigma', type=float, default=1,
                   help='Sigma value for FF encoding.')
    p.add_argument('--posenc_sdf_expbands', type=int, default=0,
                   help='Use exponential band sequence for IDR encoding?')

    # SDF Network
    p.add_argument('--init_regularized', type=int, default=0,
                   help='Use regularized weights for the sphere init?')
    p.add_argument('--fit_sphere', type=int, default=0,
                   help='Should we train for sphere only? Used to create init weights.')
    p.add_argument('--init_sphere', type=int, default=0,
                   help='Should we initialize the weights to represent unit sphere?')
    p.add_argument('--fit_plane', type=int, default=0,
                   help='Should we train for plane only? Used to create init weights.')
    p.add_argument('--init_plane', type=int, default=0,
                   help='Should we initialize the weights to represent Z=0 plane?')

    # Dataset
    p.add_argument('--dataset_path', type=str, default='/home/data/',
                   help='Path to dataset folder.')
    p.add_argument('--dataset_type', type=str, default='sinesdf_static',
                   help='Dataset type [sinesdf_static].')
    p.add_argument('--dataset_name', type=str, default='dtu',
                   help='Dataset name [dtu|nlr|shapenet]')
    p.add_argument('--world_pcd_path', type=str, default='',
                   help='Alternative path to PCD to use instead of the dataset.')
    p.add_argument('--load_pcd', type=int, default=1,
                   help='Should we load PCD for training or testing?')
    p.add_argument('--use_pcd', type=int, default=1,
                   help='Should we use PCD for training?')
    p.add_argument('--load_images', type=int, default=1,
                   help='Should we load images for training or testing?')
    p.add_argument('--work_radius', type=float, default=0.99,
                   help='To how large sphere to scale the model?')
    p.add_argument('--scene_radius_scale', type=float, default=1.0,
                   help='Scale the radius estimated from camera intersection?')
    p.add_argument('--scene_normalization', type=str, default='cache,yaml,pcd,camera',
                   help='Sets prefered space normalization mode order.')
    p.add_argument('--reference_view', type=int, default=-1,
                   help='Which view to use for RT preview? Use -1 for mid view.')
    p.add_argument('--test_views', type=str, default='',
                   help='Comma separated list of views to hold out for test purposes. Zero based indices.')
    p.add_argument('--randomize_cameras', type=int, default=0,
                   help='Should I add noise to camera poses?')

    p.add_argument('--load_im_scale', type=float, default=1.0,
                   help="Scale factor for the image training resolution. Changes the base of other scales.")
    p.add_argument('--im_scale', type=float, default=1.0,
                   help="Scale factor for the image render resolution. Only affects tests/eval/summaries.")
    p.add_argument("--color_loss", type=str, default='l2',
                   help='Which loss to use for color: l1|l2|smooth_l1')

    p.add_argument('--precomputed_3D_point_buffers', type=str, default=None,
                   help='Location of precomputed 3D position buffers.')
    p.add_argument('--save_3D_point_buffers', type=int, default=0,
                   help='Save the precomputed 3D position buffers for quicker load next run.')
    p.add_argument('--load_3D_point_buffers', type=int, default=0,
                   help='Load the precomputed 3D position buffers at directory.')

    # Ray-tracing.
    p.add_argument('--rt_bidirectional', type=int, default=1,
                   help='Use bidirectional ray tracing?.')
    p.add_argument('--rt_num_steps', type=int, default=16,
                   help='Number of steps for each ray.')
    p.add_argument('--rt_num_section_steps', type=int, default=100,
                   help='Number of uniform steps for sectioning.')
    p.add_argument('--rt_num_secant_steps', type=int, default=8,
                   help='Number of steps for secant algorithm.')
    p.add_argument('--rt_num_mask_steps', type=int, default=100,
                   help='Number of uniform steps for differentiable mask.')
    p.add_argument('--rt_step_alpha', type=float, default=1.0,
                   help="Ray step length factor.")
    p.add_argument('--rt_mask_alpha', type=float, default=50.0,
                   help="The mask softness alpha from Lipman 2020 Eq. 7.")
    p.add_argument('--rt_mask_alpha_period', type=int, default=5000,
                   help="Double the rt_mask_alpha every n steps.")
    p.add_argument('--rt_mask_alpha_period_epochs', type=int, default=-1,
                   help="Double the rt_mask_alpha every n epochs.")
    p.add_argument('--rt_mask_alpha_period_max', type=int, default=5,
                   help="Double the rt_mask_alpha at most this epochs times.")
    p.add_argument('--rt_mask_loss_weight', type=float, default=0.03,
                   help="Weight on the false positive rays mask loss.")

    # Parameters. Train which decoder?
    p.add_argument('--train_decoder_sdf', type=int, default=1,
                   help='Optimize SDF decoder?')
    p.add_argument('--train_image_encoder', type=int, default=1,
                   help='Optimize image encoder?')
    p.add_argument('--train_feature_decoder', type=int, default=1,
                   help='Optimize feature decoder?')
    p.add_argument('--train_feature_blending', type=int, default=1,
                   help='Optimize feature blender?')

    # Losses. They are always computed (if possible) but
    # can be left-out of the optimization.
    p.add_argument('--opt_sdf_onsurface', type=int, default=1,
                   help='Optimize On-surface SDF == 0')
    p.add_argument('--opt_sdf_offsurface', type=int, default=1,
                   help='Optimize Off-surface |SDF| > 0')
    p.add_argument('--opt_sdf_normal', type=int, default=1,
                   help='Optimize On-surface normal == GT')
    p.add_argument('--opt_sdf_eikonal', type=int, default=1,
                   help='Optimize ||Grad SDF|| = 1')
    p.add_argument('--opt_sdf_eikonal_w', type=float, default=1.0,
                   help='Weight for Optimize ||Grad SDF|| = 1')
    p.add_argument('--opt_sdf_lapl', type=int, default=0,
                   help='Should I use laplacian constraint? Minimizes SDF laplacian.')
    p.add_argument('--opt_sdf_direct', type=int, default=0, help='Directly optimize SDF.')
    p.add_argument('--opt_sdf_curvature', type=float, default=0,
                   help='Minimize sdf curvature?')
    p.add_argument('--opt_rays_sdf_curvature', type=float, default=0,
                   help='Minimize sdf curvature on the surface?')

    # Additional loss options.
    p.add_argument('--loss_eikonal_metric', type=str, default='l1',
                   help='Metric used for the eikonal loss l1|l2')
    p.add_argument('--opt_render_shape', type=int, default=0,
                   help='Optimize SDF shape using the view image raytracing loss?')
    p.add_argument('--opt_render_softmask', type=int, default=0,
                   help='Optimize the contour mask?')
    p.add_argument('--regularize_weights_sdf', type=float, default=0.0,
                   help='Minimize weights in the MLP?')
    p.add_argument('--opt_perceptual_loss', type=int, default=0,
                   help='Use perceptual loss on rendered target images')

    # Meta learning.
    p.add_argument('--num_meta_steps', type=int, default=64,
                   help='How many optimization steps to take in meta learning inner loop.')
    p.add_argument('--meta_lr_type', type=str, default='static',
                   help='How to treat the meta-lr for each parameter.')
    p.add_argument('--meta_lr', type=float, default=1e-1,
                   help='Meta-learning rate for parameters')
    p.add_argument('--meta_first_order', type=int, default=0,
                   help='First order approximation for MAML algorithm?')
    p.add_argument('--meta_algorithm', type=str, default='reptile',
                   help='Type of meta-learning algorithm [reptile|maml].')
    p.add_argument('--meta_lr_decay_factor', type=float, default=0.5, help='How much to decay LR.')
    p.add_argument('--meta_lr_decay_steps', type=int, default=0, help='How often to decay LR.')

    return p


def get_sdf_decoder(opt, SDF_datasets: list):
    # Losses.
    if not opt.opt_sdf_eikonal:
        opt.opt_sdf_eikonal_w = 0.0

    # RayBuilder - only needed for 2D training.
    ray_builders = []
    for i, dataset in enumerate(SDF_datasets):
        if dataset is not None and dataset.dataset_img is not None:
            ray_builder = RayBuilder(opt, dataset.dataset_img, dataset.model_matrix)
            ray_builders.append({'dataset_num': i,
                                 'ray_builder': ray_builder})

    # Define the model.
    model = modules_sdf.SDFIBRNet(opt, ray_builder=ray_builders)
    model.to(opt.device)
    return model


def get_meta_sdf_decoder(opt, model: modules_sdf.SDFIBRNet, meta_loss):

    # Define the meta wrapper on the module
    if opt.meta_algorithm == 'reptile':
        meta_model = meta_modules.Reptile(opt, hypo_module=model, loss=meta_loss)
    else:
        print('Not an implemented meta learning algorithm!')
        raise Exception('Invalid meta learning algorithm')
    meta_model.to(opt.device)
    return meta_model


def get_datasets(opt):
    """
    Gets the dataset.
    """
    if opt.dataset_name == 'nlr':
        opt.WITHHELD_VIEWS = []
        opt.TRAIN_VIEWS = [16, 17, 18, 19, 20, 21]
    elif opt.dataset_name == 'dtu':
        opt.WITHHELD_VIEWS = []
        opt.TRAIN_VIEWS = opt.TRAIN_VIEWS = list(set(list(range(0, 49))) - set(opt.WITHHELD_VIEWS))

    datasets = []
    for dataset_path in os.listdir(Path(opt.dataset_path)):
        dp = os.path.join(Path(opt.dataset_path), dataset_path)
        datasets.append(dataio_sdf.DatasetSDF(Path(dp), opt))

    return datasets


def get_meta_dataset(opt, dataset):
    """
    Returns the dataset wrapped in a form which can be processed by meta-learning
    """
    return dataio_meta.MetaDatasetSDF(dataset, opt)


def get_latest_checkpoint_file(opt):
    """
    Gets the latest checkpoint pth file.
    """
    chck_dir = Path(opt.logging_root) / opt.experiment_name / 'checkpoints'

    # Return final if exists.
    chck_final = chck_dir / 'model_final.pth'
    if chck_final.is_file():
        return chck_final

    # Return current if exists.
    chck_current = chck_dir / 'model_current.pth'
    if chck_current.is_file():
        return chck_current

    # Find all pth files.
    if chck_dir.is_dir():
        check_files = sorted([x for x in chck_dir.iterdir() if x.stem.startswith('model') and x.suffix == '.pth'])
    else:
        check_files = []
    if check_files:
        # Return latest.
        return check_files[-1]

    # No checkpoint.
    return None


def main():

    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

    # Params.
    p = get_arg_parser()
    opt = p.parse_args()
    opt.ibr_dataset = 1

    opt.occ_threshold = 5e-4

    # Clear the stringified None.
    for k, v in vars(opt).items():
        if p.get_default(k) is None and v == 'None':
            setattr(opt, k, None)

    # Create log dir and copy the config file
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    print(f'Will log into {root_path}.')
    os.makedirs(root_path, exist_ok=True)
    f = os.path.join(root_path, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(opt)):
            attr = getattr(opt, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if opt.config_filepath is not None:
        f = os.path.join(root_path, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(opt.config_filepath, 'r').read())

    if opt.restart:
        print(f'Deleting previous logs in {root_path}...')
        common_utils.cond_rmtree(Path(root_path) / 'checkpoints')
        common_utils.cond_rmtree(Path(root_path) / 'summaries')
        common_utils.cond_rmtree(Path(root_path) / 'verbose')

    # Dataset.
    sdf_datasets = get_datasets(opt)
    meta_dataset = get_meta_dataset(opt, sdf_datasets)
    dataloader = DataLoader(meta_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # Model.
    model = get_sdf_decoder(opt, sdf_datasets)
    with (Path(root_path) / 'model.txt').open('w') as file:
        file.write(f'{model}')

    # Partial checkpoints.
    if opt.checkpoint_sdf and Path(opt.checkpoint_sdf).is_file():
        model.load_checkpoint(opt.checkpoint_sdf, load_sdf=True, load_flow=False, load_poses=opt.load_model_poses)
    if opt.checkpoint_img_encoder and Path(opt.checkpoint_img_encoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_encoder, load_img_encoder=True, load_img_decoder=True)
    if opt.checkpoint_img_decoder and Path(opt.checkpoint_img_decoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_decoder, load_img_decoder=True)
    if opt.checkpoint_aggregation_mlp and Path(opt.checkpoint_aggregation_mlp).is_file():
        model.load_checkpoint(opt.checkpoint_aggregation_mlp, load_aggregation=True)

    # Resume?
    if opt.checkpoint_path:
        if not os.path.isfile(opt.checkpoint_path):
            raise RuntimeError(f"Could not find checkpoint {opt.checkpoint_path}.")
        checkpoint_file = Path(opt.checkpoint_path)
    else:
        checkpoint_file = get_latest_checkpoint_file(opt)
    if opt.resume and checkpoint_file and not (opt.fit_sphere or opt.fit_plane):
        print(f'Loading checkpoint from {checkpoint_file}...')
        model.load_checkpoint(checkpoint_file, strict=opt.checkpoint_strict)
    else:
        print('Starting training from scratch...')

    # loss_fn = partial(loss_functions.loss_sdf_ibr, opt)
    # summary_fn = partial(utils_meta.write_meta_ibr_summary, opt, meta_dataset)
    loss_fn = partial(loss_functions.loss_sdf_ibr_mult, opt)
    summary_fn = partial(utils_meta.write_meta_ibr_summary_mult, opt, meta_dataset)

    # MAML / REPTILE.
    meta_model = get_meta_sdf_decoder(opt, model, loss_fn)

    # Define optimizer.
    if opt.feature_type == "rgb":
        opt.train_feature_decoder = 0
        opt.train_image_encoder = 0

    # Define optimizer.
    params = []
    if opt.train_decoder_sdf:
        params += [{'params': meta_model.hypo_module.decoder_sdf.parameters(), 'name': 'sdf'}]
    if opt.train_feature_decoder:
        params += [{'params': meta_model.hypo_module.dec_net.parameters(), 'name': 'image_dec'}]
    if opt.train_image_encoder:
        params += [{'params': meta_model.hypo_module.enc_net.parameters(), 'name': 'image_enc'}]
    if opt.train_feature_blending and getattr(meta_model.hypo_module, 'agg_net', False):
        params += [{'params': meta_model.hypo_module.agg_net.parameters(), 'name': 'agg'}]
    optimizer = torch.optim.Adam(params=params, lr=opt.lr)

    meta_model.opt.clip_gradients = True
    # meta_model.opt.clip_gradients = False
    training.train_meta(meta_model=meta_model, meta_dataloader=dataloader, epochs=opt.num_epochs,
                        lr=opt.meta_lr, steps_til_summary=opt.steps_til_summary,
                        epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                        clip_grad=True, optim=optimizer, verbose_record_file=opt.load_verbose_record)


if __name__ == "__main__":
    main()
