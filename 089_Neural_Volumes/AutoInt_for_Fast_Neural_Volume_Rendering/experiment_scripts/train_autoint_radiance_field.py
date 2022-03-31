# Enable import from parent package
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio
import utils
import training
import loss_functions
import modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
import re

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# Experiment & I/O general properties
p.add_argument('--dataset', type=str, default='blender', choices=['blender', 'deepvoxels', 'llff'],
               help='which dataset to use')
p.add_argument('--experiment_name', type=str, default=None,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
p.add_argument('--nerf_dataset_path', type=str, default='../data/nerf_synthetic/lego/',
               help='path to directory where blender dataset is stored')
p.add_argument('--dv_dataset_path', type=str, default='../data/deepvoxel/globe/',
               help='path to directory where deepvoxels dataset is stored')
p.add_argument('--llff_dataset_path', type=str, default='../data/nerf_llff_data/fern/',
               help='path to directory where llff dataset is stored')
p.add_argument('--resume', nargs=2, type=str, default=None,
               help='resume training, specify path to directory where model is stored.')
p.add_argument('--render_model', nargs=2, type=str, default=None,
               help='Render out a trajectory, specify path to directory where model is stored.'
                    + 'specify as <dir> <epoch>')
p.add_argument('--render_select_idx', type=str, default=None,
               help='give frame number to be rendered.')
p.add_argument('--render_output', type=str, default=None,
               help='render output filename')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

# GPU & other computing properties
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')
p.add_argument('--chunk_size_train', type=int, default=1024,
               help='max chunk size to process data during training')
p.add_argument('--chunk_size_eval', type=int, default=512,
               help='max chunk size to process data during eval')
p.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers.')

# Learning properties
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--clip_grad', action='store_true', help='clips the gradients during learning')
p.add_argument('--batch_size', type=int, default=1)

# Network architecture properties
p.add_argument('--hidden_features', type=int, default=128)
p.add_argument('--hidden_layers', type=int, default=6)
p.add_argument('--activation', type=str, default='sine', choices=['sine', 'swish'],
               help='activation to use')
p.add_argument('--use_piecewise_model', action='store_true',
               help='If true, use the piecewise forward model for fast inference instead of continuous.')
p.add_argument('--normalize_pe', action='store_true',
               help='If true, add a normalization to the positional encoding.')
p.add_argument('--num_cuts', type=int, default=32,
               help='Number of cuts in the piecewise linear model')
p.add_argument('--use_grad', action='store_true', default=False,
               help='fit the gradient of the network during training')

# Nerf Properties
p.add_argument('--img_size', type=int, default=128,
               help='image resolution to train on (assumed symmetric)')
p.add_argument('--samples_per_ray', type=int, default=128,
               help='samples to evaluate along each ray')
p.add_argument('--samples_per_view', type=int, default=1024,
               help='samples to evaluate along each view')
p.add_argument('--use_sampler', action='store_true', default=False,
               help='use sampler at beginning of network')
p.add_argument('--use_sobol_ray_sampling', action='store_true', default=False,
               help='sample ray locations using sobol sampling')


opt = p.parse_args()

if opt.experiment_name is None and opt.render_model is None:
    p.error('--experiment_name is required.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():
    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    # Mode selected to run: training vs inference (rendering)
    if opt.render_model is not None:
        render_model()
    else:
        train()


def train(validation=True):
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    ''' Training dataset '''
    if opt.dataset == 'deepvoxels':
        dataset = dataio.DeepVoxelDataset(opt.dv_dataset_path,
                                          mode='train',
                                          resize_to=2*(opt.img_size,))
        use_ndc = False
    elif opt.dataset == 'llff':
        dataset = dataio.LLFFDataset(opt.llff_dataset_path,
                                     mode='train')
        use_ndc = True
    elif opt.dataset == 'blender':
        dataset = dataio.NerfBlenderDataset(opt.nerf_dataset_path,
                                            splits=['train'],  # which split to load: either 'train', 'val', 'test'
                                            mode='train',   # which split to train on (must be in splits)
                                            resize_to=2*(opt.img_size,),
                                            ref_rot=None, d_rot=None)
        use_ndc = False
    else:
        raise NotImplementedError('dataset not implemented')

    coords_dataset = dataio.Implicit6DMultiviewDataWrapper(dataset,
                                                           dataset.get_img_shape(),
                                                           dataset.get_camera_params(),
                                                           samples_per_ray=opt.samples_per_ray,
                                                           samples_per_view=opt.samples_per_view,
                                                           use_ndc=use_ndc)
    ''' Validation dataset '''
    if validation:
        if opt.dataset == 'deepvoxels':
            val_dataset = dataio.DeepVoxelDataset(opt.dv_dataset_path,
                                                  mode='val', idcs=dataset.val_idcs,
                                                  resize_to=2 * (opt.img_size,))
        elif opt.dataset == 'llff':
            val_dataset = dataio.LLFFDataset(opt.llff_dataset_path, mode='val')
        elif opt.dataset == 'blender':
            val_dataset = dataio.NerfBlenderDataset(opt.nerf_dataset_path,
                                                    splits=['val'],  # which split to load: either 'train', 'val', 'test'
                                                    mode='val',   # which split to train on (must be in splits)
                                                    resize_to=2*(opt.img_size,),
                                                    ref_rot=None, d_rot=None)

        val_coords_dataset = dataio.Implicit6DMultiviewDataWrapper(val_dataset,
                                                                   val_dataset.get_img_shape(),
                                                                   val_dataset.get_camera_params(),
                                                                   samples_per_ray=opt.samples_per_ray,
                                                                   samples_per_view=np.prod(val_dataset.get_img_shape()[:2]),
                                                                   num_workers=opt.num_workers,
                                                                   sobol_ray_sampling=opt.use_sobol_ray_sampling,
                                                                   use_ndc=use_ndc)

    ''' Dataloaders'''
    dataloader = DataLoader(coords_dataset, shuffle=True, batch_size=opt.batch_size,  # num of views in a batch
                            pin_memory=True, num_workers=opt.num_workers)

    if validation:
        val_dataloader = DataLoader(val_coords_dataset, shuffle=True, batch_size=1,
                                    pin_memory=True, num_workers=opt.num_workers)
    else:
        val_dataloader = None

    # get model paths
    if opt.resume is not None:
        path, epoch = opt.resume
        epoch = int(epoch)
        assert(os.path.isdir(path))
        assert opt.config is not None, 'Specify config file'

    if opt.use_sampler:
        cam_params = dataset.get_camera_params()
        sampler = modules.SamplingNet(Nt=opt.samples_per_ray, ncuts=opt.num_cuts,
                                      sampling_interval=(cam_params['near'], cam_params['far']))
    else:
        sampler = None

    add_pe_ray_samples = 10  # 10 cos + sin
    add_pe_orientations = 4  # 4 cos + sin
    nl_types = opt.activation

    model_sigma = modules.RadianceNet(out_features=1,
                                      hidden_layers=opt.hidden_layers,
                                      hidden_features=opt.hidden_features,
                                      nl=nl_types,
                                      use_grad=opt.use_grad,
                                      input_name=['ray_samples',
                                                  'ray_orientations'],
                                      input_processing_fn=modules.input_processing_fn,
                                      input_pe_params={'ray_samples': add_pe_ray_samples,
                                                       'ray_orientations': add_pe_orientations},
                                      sampler=sampler,
                                      normalize_pe=opt.normalize_pe)
    model_sigma.cuda()

    model_rgb = modules.RadianceNet(out_features=3,
                                    hidden_layers=opt.hidden_layers,
                                    hidden_features=opt.hidden_features,
                                    nl=nl_types,
                                    use_grad=opt.use_grad,
                                    input_name=['ray_samples',
                                                'ray_orientations'],
                                    input_processing_fn=modules.input_processing_fn,
                                    input_pe_params={'ray_samples': add_pe_ray_samples,
                                                     'ray_orientations': add_pe_orientations},
                                    sampler=sampler,
                                    normalize_pe=opt.normalize_pe)
    model_rgb.cuda()

    if opt.resume is not None:
        if(epoch > 0):
            model_path_sigma = path + '/checkpoints/' + f'model_sigma_epoch_{epoch:04d}.pth'
            model_path_rgb = path + '/checkpoints/' + f'model_rgb_epoch_{epoch:04d}.pth'
        else:
            model_path_sigma = path + '/checkpoints/' + 'model_sigma_current.pth'
            model_path_rgb = path + '/checkpoints/' + 'model_rgb_current.pth'
        print('Loading checkpoints')
        ckpt_dict = torch.load(model_path_sigma)
        state_dict = translate_saved_weights(ckpt_dict, model_sigma)
        model_sigma.load_state_dict(state_dict, strict=True)
        ckpt_dict = torch.load(model_path_rgb)
        state_dict = translate_saved_weights(ckpt_dict, model_rgb)
        model_rgb.load_state_dict(state_dict, strict=True)

        # load optimizers
        try:
            if (epoch > 0):
                optim_path_sigma = path + '/checkpoints/' + f'optim_sigma_epoch_{epoch:04d}.pth'
                optim_path_rgb = path + '/checkpoints/' + f'optim_rgb_epoch_{epoch:04d}.pth'
            else:
                optim_path_sigma = path + '/checkpoints/' + 'optim_sigma_current.pth'
                optim_path_rgb = path + '/checkpoints/' + 'optim_rgb_current.pth'
            resume_checkpoint = {}
            sigma_ckpt = torch.load(optim_path_sigma)
            for g in sigma_ckpt['optimizer_state_dict']['param_groups']:
                g['lr'] = opt.lr
            resume_checkpoint['sigma'] = sigma_ckpt['optimizer_state_dict']
            rgb_ckpt = torch.load(optim_path_rgb)
            for g in rgb_ckpt['optimizer_state_dict']['param_groups']:
                g['lr'] = opt.lr
            resume_checkpoint['rgb'] = rgb_ckpt['optimizer_state_dict']
            resume_checkpoint['total_steps'] = rgb_ckpt['total_steps']
            resume_checkpoint['epoch'] = rgb_ckpt['epoch']
        except FileNotFoundError:
            print('Unable to load optimizer checkpoints')
    else:
        resume_checkpoint = {}
    models = {'sigma': model_sigma,
              'rgb': model_rgb}

    # Define the loss
    loss_fn = partial(loss_functions.tomo_radiance_sigma_rgb_loss,
                      use_piecewise_model=opt.use_piecewise_model,
                      num_cuts=opt.num_cuts)
    summary_fn = partial(utils.write_tomo_radiance_summary,
                         chunk_size_eval=opt.chunk_size_eval,
                         num_views_to_disp_at_training=1,
                         use_piecewise_model=opt.use_piecewise_model,
                         num_cuts=opt.num_cuts, use_coarse_fine=False)
    chunk_lists_from_batch_fn = dataio.chunk_lists_from_batch_reduce_to_raysamples_fn

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])
    with open(os.path.join(root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        for model_name, model in models.items():
            out_file.write(model_name)
            out_file.write(str(model))

    training.train_wchunks(models=models, train_dataloader=dataloader,
                           epochs=opt.num_epochs, lr=opt.lr,
                           steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                           model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                           val_dataloader=val_dataloader,
                           chunk_lists_from_batch_fn=chunk_lists_from_batch_fn,
                           max_chunk_size=opt.chunk_size_train,
                           num_cuts=opt.num_cuts,
                           clip_grad=opt.clip_grad,
                           resume_checkpoint=resume_checkpoint)


def translate_saved_weights(ckpt_dict, model):
    """
        Keys from saved models are encoded with a unique session ID.
        We need to parse that ID out and replace it with the ID of the current model
        before attempting to load the weights
    """

    state_dict = {}
    for k, v in ckpt_dict.items():
        if 'sampler' in k:
            new_k = k
        elif 'backward_session' in k:
            new_k = re.search(r'.*_.*_', k).group(0) + str(model.backward_session.uid) + re.findall(r'\.[a-zA-Z]+', k)[-1]
        else:
            new_k = re.search(r'.*_.*_', k).group(0) + str(model.session.uid) + re.findall(r'\.[a-zA-Z]+', k)[-1]

        if v.ndim == 3:
            v = v[:1, :1, :]

        state_dict[new_k] = v
    return state_dict


def render_model():
    # get model paths
    path, epoch = opt.render_model
    epoch = int(epoch)
    assert(os.path.isdir(path))
    assert(os.path.isfile(path + '/config.ini'))

    p = configargparse.DefaultConfigFileParser()
    with open(path + '/config.ini') as f:
        args = p.parse(f)
    opt.hidden_layers = int(args['hidden_layers'])
    opt.hidden_features = int(args['hidden_features'])
    opt.use_piecewise_model = args['use_piecewise_model'] == 'true'
    opt.use_grad = args['use_grad'] == 'true'
    opt.activation = args['activation']
    opt.normalize_pe = args['normalize_pe'] == 'true'
    opt.img_size = int(args['img_size'])
    opt.num_cuts = int(args['num_cuts'])
    opt.use_sampler = args['use_sampler'] == 'true'
    opt.dataset = args['dataset']

    if opt.dataset == 'deepvoxels':
        dataset = dataio.DeepVoxelDataset(opt.dv_dataset_path,
                                          mode='test',
                                          resize_to=2*(opt.img_size,))
        use_ndc = False
    elif opt.dataset == 'llff':
        dataset = dataio.LLFFDataset(opt.llff_dataset_path,
                                     mode='test', final_render=False)
        use_ndc = True
    elif opt.dataset == 'blender':
        dataset = dataio.NerfBlenderDataset(opt.nerf_dataset_path,
                                            splits=['test'],
                                            mode='test',
                                            select_idx=opt.render_select_idx,
                                            resize_to=2*(opt.img_size,))
        use_ndc = False
    else:
        raise NotImplementedError('dataset not implemented')

    if opt.use_sampler:
        cam_params = dataset.get_camera_params()
        sampler = modules.SamplingNet(Nt=opt.samples_per_ray, ncuts=opt.num_cuts,
                                      sampling_interval=(cam_params['near'], cam_params['far']))
    else:
        sampler = None

    add_pe_ray_samples = 10  # 10 cos + sin
    add_pe_orientations = 4  # 4 cos + sin

    model_sigma = modules.RadianceNet(out_features=1,
                                      hidden_layers=opt.hidden_layers,
                                      hidden_features=opt.hidden_features,
                                      nl=opt.activation,
                                      use_grad=opt.use_grad,
                                      input_name=['ray_samples',
                                                  'ray_orientations'],
                                      input_processing_fn=modules.input_processing_fn,
                                      input_pe_params={'ray_samples': add_pe_ray_samples,
                                                       'ray_orientations': add_pe_orientations},
                                      sampler=sampler,
                                      normalize_pe=opt.normalize_pe)

    ckpt_dict = torch.load(path + '/checkpoints/' + f'model_sigma_epoch_{epoch:04d}.pth')
    state_dict = translate_saved_weights(ckpt_dict, model_sigma)
    model_sigma.load_state_dict(state_dict, strict=True)
    model_sigma.eval()
    model_sigma.cuda()

    model_rgb = modules.RadianceNet(out_features=3,
                                    hidden_layers=opt.hidden_layers,
                                    hidden_features=opt.hidden_features,
                                    nl=opt.activation,
                                    use_grad=opt.use_grad,
                                    input_name=['ray_samples',
                                                'ray_orientations'],
                                    input_processing_fn=modules.input_processing_fn,
                                    input_pe_params={'ray_samples': add_pe_ray_samples,
                                                     'ray_orientations': add_pe_orientations},
                                    sampler=sampler,
                                    normalize_pe=opt.normalize_pe)

    ckpt_dict = torch.load(path + '/checkpoints/' + f'model_rgb_epoch_{epoch:04d}.pth')
    state_dict = translate_saved_weights(ckpt_dict, model_rgb)
    model_rgb.load_state_dict(state_dict, strict=True)
    model_rgb.eval()
    model_rgb.cuda()

    models = {'sigma': model_sigma,
              'rgb': model_rgb}

    # set up dataset
    coords_dataset = dataio.Implicit6DMultiviewDataWrapper(dataset,
                                                           dataset.get_img_shape(),
                                                           dataset.get_camera_params(),
                                                           samples_per_ray=opt.samples_per_ray,
                                                           samples_per_view=np.prod(dataset.get_img_shape()[:2]),
                                                           use_ndc=use_ndc)
    coords_dataset.toggle_logging_sampling()

    if opt.render_output is None:
        output_path = path + '/render'
    else:
        output_path = opt.render_output

    utils.render_views(output_path, models, coords_dataset,
                       use_piecewise_model=opt.use_piecewise_model,
                       num_cuts=opt.num_cuts,
                       use_sampler=opt.use_sampler,
                       integral_render=True,
                       chunk_size=opt.chunk_size_train, video=False)
    sys.exit()


if __name__ == '__main__':
    main()
