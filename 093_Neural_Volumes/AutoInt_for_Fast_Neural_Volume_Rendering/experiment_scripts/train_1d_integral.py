# Enable import from parent package
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
import torch
from functools import partial

torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--activation', type=str, default='swish',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='activation to use')
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--w0', type=float, default=10)
p.add_argument('--hidden_features', type=int, default=128)
p.add_argument('--hidden_layers', type=int, default=4)
p.add_argument('--experiment_name', type=str, default='train_1d_integral',
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--lr', type=float, default=5e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=5001,
               help='Number of epochs to train for.')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

# summary options
p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

# logging options
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
p.add_argument('--model', type=str, default='siren', required=False, choices=['siren', 'nerf'],
               help='Type of model to evaluate, default is siren.')

opt = p.parse_args()

if opt.experiment_name is None and opt.render_model is None:
    p.error('--experiment_name is required.')

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():

    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    train()


def train(validation=True):
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    fn = dataio.polynomial_1
    integral_fn = dataio.polynomial_1_integral
    train_dataset = dataio.Implicit1DWrapper(range=(-1, 2),
                                             fn=fn,
                                             integral_fn=integral_fn,
                                             sampling_density=1000,
                                             train_every=250)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    if opt.activation != 'sine':
        num_pe_functions = 4  # cos + sin
    else:
        num_pe_functions = 0

    model = modules.CoordinateNet(nl=opt.activation,
                                  in_features=1,
                                  out_features=1,
                                  hidden_features=opt.hidden_features,
                                  num_hidden_layers=opt.hidden_layers,
                                  w0=opt.w0, use_grad=True,
                                  num_pe_fns=num_pe_functions,
                                  input_processing_fn=lambda x: x,
                                  grad_var='coords')
    model.cuda()

    # Define the loss
    loss_fn = loss_functions.function_mse
    summary_fn = partial(utils.write_simple_1D_function_summary, train_dataset)

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])
    with open(os.path.join(root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    training.train(model=model, train_dataloader=train_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)


if __name__ == '__main__':
    main()
