# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import logging
import os
import json
import configargparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from util.utils import make_dir


def run_main(config):
    config.global_size = config.n_nodes * config.n_gpus_per_node
    processes = []
    for rank in range(config.n_gpus_per_node):
        config.local_rank = rank
        config.global_rank = rank + config.node_rank * config.n_gpus_per_node
        print('Node rank %d, local proc %d, global proc %d' %
              (config.node_rank, config.local_rank, config.global_rank))
        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.master_address
    os.environ['MASTER_PORT'] = '%d' % config.master_port
    torch.cuda.set_device(config.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.global_rank,
                            world_size=config.global_size)
    fn(config)
    dist.barrier()
    dist.destroy_process_group()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(config):
    if config.workdir[-1] == '/':
        config.workdir = config.workdir[:-1]
        
    workdir = os.path.join(config.root, config.workdir +
                           ('_seed_%d' % config.seed))

    if config.mode == 'train':
        if config.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)

        if config.is_image:
            import run_lib
            run_lib.train(config, workdir)
        else:
            import run_lib_toy
            run_lib_toy.train(config, workdir)

    elif config.mode == 'eval':
        if os.path.exists(workdir):
            if config.global_rank == 0:
                if config.eval_folder is None:
                    raise ValueError('Need to set eval folder.')
                eval_dir = os.path.join(workdir, config.eval_folder)
                make_dir(eval_dir)
                gfile_stream = open(os.path.join(eval_dir, 'stdout.txt'), 'w')
                set_logger(gfile_stream)

            if config.is_image:
                import run_lib
                run_lib.evaluate(config, workdir)
            else:
                import run_lib_toy
                run_lib_toy.evaluate(config, workdir)
        else:
            raise ValueError('No experiment to evaluate.')

    elif config.mode == 'continue':
        if os.path.exists(workdir):
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')
            set_logger(gfile_stream)

            if config.is_image:
                import run_lib
                run_lib.train(config, workdir)
            else:
                import run_lib_toy
                run_lib_toy.train(config, workdir)

    else:
        raise ValueError('Mode not recognized.')


if __name__ == '__main__':
    p = configargparse.ArgParser()
    p.add('-cc', is_config_file=True)
    p.add('-sc', is_config_file=True)

    p.add('--root')
    p.add('--workdir', required=True)
    p.add('--eval_folder', default=None)
    p.add('--mode', choices=['train', 'eval', 'continue'], required=True)
    p.add('--cont_nbr', type=int, default=None)
    p.add('--checkpoint', default=None)

    p.add('--n_gpus_per_node', type=int, default=1)
    p.add('--n_nodes', type=int, default=1)
    p.add('--node_rank', type=int, default=0)
    p.add('--master_address', default='127.0.0.1')
    p.add('--master_port', type=int, default=6020)
    p.add('--distributed', action='store_false')

    p.add('--overwrite', action='store_true')

    p.add('--seed', type=int, default=0)

    # Data
    p.add('--dataset')
    p.add('--is_image', action='store_true')
    p.add('--image_size', type=int)
    p.add('--center_image', action='store_true')
    p.add('--image_channels', type=int)
    p.add('--data_dim', type=int)  # Dimension of non-image data
    p.add('--data_location', default=None)

    # SDE
    p.add('--sde')
    p.add('--beta_type')
    # Linear beta params
    p.add('--beta0', type=float)
    p.add('--beta1', type=float)
    # CLD params
    p.add('--m_inv', type=float)
    p.add('--gamma', type=float)
    p.add('--numerical_eps', type=float)

    # Optimization
    p.add('--optimizer')
    p.add('--learning_rate', type=float)
    p.add('--weight_decay', type=float)
    p.add('--grad_clip', type=float)

    # Objective
    p.add('--cld_objective', choices=['dsm', 'hsm'], default='hsm')
    p.add('--loss_eps', type=float)
    p.add('--weighting', choices=['likelihood',
          'reweightedv1', 'reweightedv2'])

    # Model
    p.add('--name')
    p.add('--ema_rate', type=float)
    p.add('--normalization')
    p.add('--nonlinearity')
    p.add('--n_channels', type=int)
    p.add('--ch_mult')
    p.add('--n_resblocks', type=int)
    p.add('--attn_resolutions')
    p.add('--resamp_with_conv', action='store_true')
    p.add('--use_fir', action='store_true')
    p.add('--fir_kernel')
    p.add('--skip_rescale', action='store_true')
    p.add('--resblock_type')
    p.add('--progressive')
    p.add('--progressive_input')
    p.add('--progressive_combine')
    p.add('--attention_type')
    p.add('--init_scale', type=float)
    p.add('--fourier_scale', type=int)
    p.add('--conv_size', type=int)
    p.add('--dropout', type=float)
    p.add('--mixed_score', action='store_true')
    p.add('--embedding_type', choices=['fourier', 'positional'])

    # Training
    p.add('--training_batch_size', type=int)
    p.add('--testing_batch_size', type=int)
    p.add('--sampling_batch_size', type=int)
    p.add('--n_train_iters', type=int)
    p.add('--n_warmup_iters', type=int)
    p.add('--snapshot_freq', type=int)
    p.add('--log_freq', type=int)
    p.add('--eval_freq', type=int)
    p.add('--likelihood_freq', type=int)
    p.add('--fid_freq', type=int)
    p.add('--eval_threshold', type=int, default=1)
    p.add('--likelihood_threshold', type=int, default=1)
    p.add('--snapshot_threshold', type=int, default=1)
    p.add('--fid_threshold', type=int, default=1)
    p.add('--fid_samples_training', type=int)
    p.add('--n_eval_batches', type=int)
    p.add('--n_likelihood_batches', type=int)
    p.add('--autocast_train', action='store_true')
    p.add('--save_freq', type=int, default=None)
    p.add('--save_threshold', type=int, default=1)

    # Sampling
    p.add('--sampling_method', choices=['ode', 'em', 'sscs'], default='ode')
    p.add('--sampling_solver', default='scipy_solver')
    p.add('--sampling_solver_options',
          type=json.loads, default={'solver': 'RK45'})
    p.add('--sampling_rtol', type=float, default=1e-5)
    p.add('--sampling_atol', type=float, default=1e-5)
    p.add('--sscs_num_stab', type=float, default=0.)
    p.add('--denoising', action='store_true')
    p.add('--n_discrete_steps', type=int)
    p.add('--striding', choices=['linear', 'quadratic'], default='linear')
    p.add('--sampling_eps', type=float)

    # Likelihood
    p.add('--likelihood_solver', default='scipy_solver')
    p.add('--likelihood_solver_options',
          type=json.loads, default={'solver': 'RK45'})
    p.add('--likelihood_rtol', type=float, default=1e-5)
    p.add('--likelihood_atol', type=float, default=1e-5)
    p.add('--likelihood_eps', type=float, default=1e-5)
    p.add('--likelihood_hutchinson_type',
          choices=['gaussian', 'rademacher'], default='rademacher')

    # Evaluation
    p.add('--ckpt_file')
    p.add('--eval_sample', action='store_true')
    p.add('--autocast_eval', action='store_true')
    p.add('--eval_loss', action='store_true')
    p.add('--eval_fid', action='store_true')
    p.add('--eval_likelihood', action='store_true')
    p.add('--eval_fid_samples', type=int, default=50000)
    p.add('--eval_jacobian_norm', action='store_true')
    p.add('--eval_iw_likelihood', action='store_true')
    p.add('--eval_density', action='store_true')
    p.add('--eval_density_npts', type=int, default=101)
    p.add('--eval_sample_hist', action='store_true')
    p.add('--eval_hist_samples', type=int, default=100000)
    p.add('--eval_loss_variance', action='store_true')
    p.add('--eval_loss_variance_images', type=int, default=1)
    p.add('--eval_sample_samples', type=int, default=1)

    p.add('--eval_seed', type=int, default=0)

    config = p.parse_args()
    run_main(config)
