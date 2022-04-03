# Enable import from parent package
import sys
import os
from functools import partial
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio
import utils
import training
import loss_functions
import modules
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# General training options #lr=1e-5
p.add_argument('--batch_size', type=int, default=4096)
p.add_argument('--experiment_name', type=str, default='train_sparse_tomography',
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=50000,
               help='Number of epochs to train for.')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')
p.add_argument('--subsample', type=int, default=4,
               help='subsampling factor')
p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--activation', type=str, default='swish', choices=['sine', 'swish', 'relu', 'softplus'],
               help='neural network activation to be used')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')

p.add_argument('--use_grad',   action='store_true', default=False, help='train with grad network or not')
p.add_argument('--evaluate',   action='store_true', default=False, help='evaluate the model')
p.add_argument('--rho_res',   type=int, default=64, help='Rho resolution of the sinogram (projection width)')
p.add_argument('--theta_res', type=int, default=64, help='Theta resolution of the sinogram (projection angles)')
p.add_argument('--mc_res',   type=int, default=64, help='Monte carlo samples')
p.add_argument('--output_dir', type=str, default='./', help='Where to store results')
opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def input_processing_fn(model_input):
    model_input['coords'] = torch.cat((model_input['rho'],
                                       model_input['theta'],
                                       model_input['t']), dim=-1)
    return model_input


def train_parallel_beam():
    rho_resolution = opt.rho_res
    theta_resolution = opt.theta_res
    dataset = dataio.SheppLoganPhantomRadonTransformed(rho_resolution=rho_resolution,
                                                       theta_resolution=theta_resolution)

    coord_dataset = dataio.Implicit2DRadonTomoWrapper(dataset,
                                                      mc_resolution=opt.mc_res,
                                                      rho_resolution=rho_resolution,
                                                      theta_resolution=theta_resolution,
                                                      subsampling_factor=opt.subsample)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    # Define the model.
    model = modules.CoordinateNet(in_features=3, out_features=1, nl=opt.activation,
                                  hidden_features=128, num_hidden_layers=4, num_pe_fns=10,
                                  grad_var='t',
                                  use_grad=opt.use_grad,
                                  input_processing_fn=input_processing_fn)
    model.cuda()

    loss_fn = loss_functions.tomography2D
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    summary_fn = partial(utils.write_tomography2D_summary, root_path, 'standard', None, (rho_resolution, theta_resolution))

    utils.cond_mkdir(root_path)

    # Save command-line parameters log directory.
    with open(os.path.join(root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)


if __name__ == '__main__':
    train_parallel_beam()
