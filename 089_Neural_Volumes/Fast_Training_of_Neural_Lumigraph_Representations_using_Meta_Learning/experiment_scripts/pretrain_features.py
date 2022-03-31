"""
Pre-trains the encoder and decoder for nlr++ model
"""
# Enable import from parent package
from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

import utils.common_utils as common_utils

import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import configargparse

import data_processing.datasets.dataio_features as dataio_features
import utils.utils_ibr as utils_ibr

import loss_functions
import modules_unet

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time

import numpy as np


def get_arg_parser():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    # Save/resume.
    p.add_argument('--logging_root', type=str, default='./logs_pretrain', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                   help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help='Checkpoint to trained model. Latest used as default.')
    p.add_argument('--checkpoint_img_encoder', type=str, default=None,
                   help='Checkpoint to only use for Image Encoder.')
    p.add_argument('--checkpoint_img_decoder', type=str, default=None,
                   help='Checkpoint to only use for Image Decoder.')
    p.add_argument('--resume', type=int, default=1,
                   help='Resume from previous checkpoint?')
    p.add_argument('--restart', type=int, default=1,
                   help='Remove all prev checkpoints and summaries in the log dir?')

    # General training options
    p.add_argument('--device', type=str, default='cuda', help='Device to use.')
    p.add_argument('--batch_size', type=int, default=16, help='Number of points for 3D supervision')
    p.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for.')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--lr_decay_factor', type=float, default=0.5, help='How omuch to decay LR.')
    p.add_argument('--lr_encdec_decay_steps', type=int, default=0, help='How often to decay LR.')

    p.add_argument('--epochs_til_ckpt', type=int, default=2,
                   help='Time interval in epochs until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=1000,
                   help='Time interval in steps until tensorboard summary is saved.')

    # CNN Models
    p.add_argument('--model_image_encoder_depth', type=int, default=3,
                   help='Depth of the ResNet used for encoding individual images.')
    p.add_argument('--model_image_encoder_features', type=int, default=16,
                   help='Size of output features of each image.')
    p.add_argument('--model_image_decoder_depth', type=int, default=2,
                   help='Depth of the UNet used for decoding an image from feature.')
    p.add_argument('--convolution_type', type=str, default='standard',
                   help='Convolution type [standard|partial]')

    # Dataset
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Path to dataset folder.')
    p.add_argument('--dataset_type', type=str, default='DatasetFlyingChairs2',
                   help='Dataset type [DatasetFlyingChairs2|].')
    p.add_argument('--im_scale', type=float, default=1.0,
                   help="Scale factor for the image render resolution. Only affects tests/eval/summaries.")

    # Losses
    p.add_argument("--color_loss", type=str, default='l2',
                   help='Which loss to use for color: l1|l2|smooth_l1')
    p.add_argument("--input_image_loss", type=int, default=1,
                   help='Apply loss to encoding and decoding the input img0')

    # Parameters. Train which decoder?
    p.add_argument('--train_image_encoder', type=int, default=1,
                   help='Optimize image encoder?')
    p.add_argument('--train_feature_decoder', type=int, default=1,
                   help='Optimize feature decoder?')

    return p


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

    # No checkpoint.
    return None


def get_dataset(opt):
    if opt.dataset_type == "DatasetFlyingChairs2":
        dataset = dataio_features.DatasetFlyingChairs2(opt.dataset_path, opt)
    else:
        raise RuntimeError(f"Dataset type {opt.dataset_type} not yet implemented.")
    return dataset


def save_checkpoint(checkpoints_dir, model, optim, epoch_or_totalsteps):
    checkpoints_dir = Path(checkpoints_dir)

    suffix = f'{epoch_or_totalsteps[1]}_{epoch_or_totalsteps[0]:08d}'
    torch.save(model.state_dict(), checkpoints_dir / f'model_{suffix}.pth')
    torch.save(optim.state_dict(), checkpoints_dir / f'optim_{suffix}.pth')


def train(opt, model, train_dataloader, epochs, steps_til_summary, epochs_til_checkpoint,
          model_dir, loss_fn, summary_fn, optim):
    os.makedirs(model_dir, 0o777, True)
    summaries_dir = os.path.join(model_dir, 'summaries')
    common_utils.cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    common_utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    device = opt.device
    model.to(device)

    print(f'Will train for {epochs} epochs with {len(train_dataloader)} iterations per epoch.')
    total_steps = 0
    epoch_0 = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(epoch_0, epochs):
            if epoch % epochs_til_checkpoint == 0 and epoch != 0:
                save_checkpoint(checkpoints_dir, model, optim, (epoch, 'epoch'))

            for step, model_input in enumerate(train_dataloader):
                start_time = time.time()

                # To GPU.
                for k, v in model_input.items():
                    v = v.to(device)
                    if v.dtype == torch.float:
                        v = v.requires_grad_(True)
                    model_input[k] = v

                # Forward.
                model_output = model(model_input)

                # Compute losses.
                losses = loss_fn(model_output, model_input)

                # Sum only active losses for optimization.
                train_loss = 0.
                for loss_name, (loss, loss_enabled) in losses.items():
                    single_loss = loss.mean()
                    if torch.isnan(single_loss).any().item():
                        print('We have NAN in loss!!!!')
                        import pdb
                        pdb.set_trace()
                        raise Exception('NaN in loss!')

                    writer.add_scalar(loss_name, single_loss, total_steps)

                    if loss_enabled:
                        # Sum only active losses.
                        train_loss += single_loss

                # Log summary loss.
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                # Summarize?
                if total_steps % steps_til_summary == 0:
                    save_checkpoint(checkpoints_dir, model, optim, (total_steps, 'iter'))
                    summary_fn(model, model_input, model_output, writer, total_steps)

                # Backward.
                optim.zero_grad()
                train_loss.backward()

                grads_isnan = {k: torch.isnan(x.grad).any().item()
                               for k, x in model.named_parameters() if x.grad is not None}
                if np.any(list(grads_isnan.values())):
                    print('We have NAN in gradients!!!!')
                    import pdb
                    pdb.set_trace()
                    raise Exception('NaN in gradients!')

                optim.step()

                params_isnan = [torch.isnan(x).any().item() for x in model.parameters()]
                if np.any(params_isnan):
                    print('We have NAN in parameters!!!!')
                    import pdb
                    pdb.set_trace()
                    raise Exception('NaN in parameters!')

                pbar.update(1)

                if total_steps % 100 == 0:
                    tqdm.write("Epoch %d/%d, Total loss %0.6f, iteration time %0.6f" %
                               (epoch, epochs, train_loss.item(), time.time() - start_time))
                total_steps += 1

        # Final checkpoint.
        save_checkpoint(checkpoints_dir, model, optim, (epochs, 'epoch'))


def main():
    # Params
    p = get_arg_parser()
    opt = p.parse_args()

    torch.manual_seed(2)

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

    # Dataset
    print(f'Creating dataset {opt.dataset_type}')
    dataset = get_dataset(opt)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    # Model
    print(f'Creating model.')
    model = modules_unet.pretrain_enc_dec(opt)
    with (Path(root_path) / 'model.txt').open('w') as file:
        file.write(f'{model}')

    if opt.checkpoint_img_encoder and Path(opt.checkpoint_img_encoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_encoder, load_img_encoder=True, load_img_decoder=True)
    if opt.checkpoint_img_decoder and Path(opt.checkpoint_img_decoder).is_file():
        model.load_checkpoint(opt.checkpoint_img_decoder, load_img_decoder=True)

    # Resume?
    if opt.checkpoint_path:
        if not os.path.isfile(opt.checkpoint_path):
            raise RuntimeError(f"Could not find checkpoint {opt.checkpoint_path}.")
        checkpoint_file = Path(opt.checkpoint_path)
    else:
        checkpoint_file = get_latest_checkpoint_file(opt)
    if opt.resume and checkpoint_file:
        print(f'Loading checkpoint from {checkpoint_file}...')
        model.load_state_dict(torch.load(checkpoint_file))
    else:
        print('Starting training from scratch...')

    # Define the loss
    print(f'Defining loss, summary, and optimizers.')
    loss_fn = partial(loss_functions.loss_pretrain_features, opt)
    summary_fn = partial(utils_ibr.write_pretrain_features_summary, opt, dataset)

    params = []
    if opt.train_image_encoder:
        params += [{'params': model.enc_net.parameters(), 'name': 'image_enc'}]
    if opt.train_feature_decoder:
        params += [{'params': model.dec_net.parameters(), 'name': 'image_dec'}]
    optimizer = torch.optim.Adam(params=params, lr=opt.lr)

    print(f'Training')
    train(opt=opt, model=model, train_dataloader=dataloader, epochs=opt.num_epochs,
          steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
          model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, optim=optimizer)


if __name__ == "__main__":
    main()