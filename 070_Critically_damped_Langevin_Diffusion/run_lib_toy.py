# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import logging
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import cm

from models import mlp
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from util.utils import make_dir, get_optimizer, optimization_manager, set_seeds, compute_eval_loss, compute_non_image_likelihood, broadcast_params, reduce_tensor, build_beta_fn, build_beta_int_fn
from util.checkpoint import save_checkpoint, restore_checkpoint
import losses
import sde_lib
import sampling
import likelihood
from util.toy_data import inf_data_gen


def train(config, workdir):
    ''' Main training script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    global_size = config.global_size

    if config.mode == 'train':
        set_seeds(global_rank, config.seed)
    elif config.mode == 'continue':
        set_seeds(global_rank, config.seed + config.cont_nbr)
    else:
        raise NotImplementedError('Mode %s is unknown.' % config.mode)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    # Setting up all necessary folders
    sample_dir = os.path.join(workdir, 'samples')
    tb_dir = os.path.join(workdir, 'tensorboard')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    likelihood_dir = os.path.join(workdir, 'likelihood')

    if global_rank == 0:
        logging.info(config)
        if config.mode == 'train':
            make_dir(sample_dir)
            make_dir(tb_dir)
            make_dir(checkpoint_dir)
            make_dir(likelihood_dir)
        writer = tensorboard.SummaryWriter(tb_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.sde)

    # Creating the score model
    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())  # Sync all parameters
    score_model = DDP(score_model, device_ids=[local_rank])

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    if global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, score_model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    dist.barrier()

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    if config.mode == 'continue':
        if config.checkpoint is None:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        else:
            ckpt_path = os.path.join(checkpoint_dir, config.checkpoint)

        if global_rank == 0:
            logging.info('Loading model from path: %s' % ckpt_path)
        dist.barrier()

        state = restore_checkpoint(ckpt_path, state, device=config.device)

    num_total_iter = config.n_train_iters

    if global_rank == 0:
        logging.info('Number of total iterations: %d' % num_total_iter)
    dist.barrier()

    optimize_fn = optimization_manager(config)
    train_step_fn = losses.get_step_fn(True, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Starting training at step %d' % step)
    dist.barrier()

    if config.mode == 'continue':
        config.eval_threshold = max(step + 1, config.eval_threshold)
        config.snapshot_threshold = max(step + 1, config.snapshot_threshold)
        config.likelihood_threshold = max(
            step + 1, config.likelihood_threshold)
        config.save_threshold = max(step + 1, config.save_threshold)

    while step < num_total_iter:
        if step % config.likelihood_freq == 0 and step >= config.likelihood_threshold:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            mean_nll = compute_non_image_likelihood(
                config, sde, state, likelihood_fn, inf_data_gen, step=step, likelihood_dir=likelihood_dir)
            ema.restore(score_model.parameters())

            if global_rank == 0:
                logging.info('Mean Nll at step: %d: %.5f' %
                             (step, mean_nll.item()))
                writer.add_scalar('mean_nll', mean_nll.item(), step)

                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % step)
                if not os.path.isfile(checkpoint_file):
                    save_checkpoint(checkpoint_file, state)
            dist.barrier()

        if (step % config.snapshot_freq == 0 or step == num_total_iter) and global_rank == 0 and step >= config.snapshot_threshold:
            logging.info('Saving snapshot checkpoint.')
            save_checkpoint(os.path.join(
                checkpoint_dir, 'snapshot_checkpoint.pth'), state)

            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            x, v, nfe = sampling_fn(score_model)
            ema.restore(score_model.parameters())

            logging.info('NFE snapshot at step %d: %d' % (step, nfe))
            writer.add_scalar('nfe', nfe, step)

            this_sample_dir = os.path.join(sample_dir, 'iter_%d' % step)
            make_dir(this_sample_dir)

            plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], s=3)
            plt.savefig(os.path.join(this_sample_dir,
                        'sample_rank_%d.png' % global_rank))
            plt.close()

            if config.sde == 'cld':
                np.save(os.path.join(this_sample_dir, 'sample_x'), x.cpu())
                np.save(os.path.join(this_sample_dir, 'sample_v'), v.cpu())
            else:
                np.save(os.path.join(this_sample_dir, 'sample'), x.cpu())
        dist.barrier()

        if config.save_freq is not None:
            if step % config.save_freq == 0 and step >= config.save_threshold:
                if global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % step)
                    if not os.path.isfile(checkpoint_file):
                        save_checkpoint(checkpoint_file, state)
                dist.barrier()

        # Training
        start_time = time.time()

        x = inf_data_gen(config.dataset, config.training_batch_size).to(
            config.device)
        loss = train_step_fn(state, x)

        if step % config.log_freq == 0:
            loss = reduce_tensor(loss, global_size)
            if global_rank == 0:
                logging.info('Iter %d/%d Loss: %.4f Time: %.3f' % (step + 1,
                             config.n_train_iters, loss.item(), time.time() - start_time))
                writer.add_scalar('training_loss', loss, step)
            dist.barrier()

        step += 1

    if global_rank == 0:
        logging.info('Finished after %d iterations.' % config.n_train_iters)
        logging.info('Saving final checkpoint.')
        save_checkpoint(os.path.join(
            checkpoint_dir, 'final_checkpoint.pth'), state)
    dist.barrier()


def evaluate(config, workdir):
    ''' Main evaluation script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    set_seeds(global_rank, config.seed + config.eval_seed)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    eval_dir = os.path.join(workdir, config.eval_folder)
    sample_dir = os.path.join(eval_dir, 'samples')
    if global_rank == 0:
        logging.info(config)
        make_dir(sample_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.vpsde)

    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())
    score_model = DDP(score_model, device_ids=[local_rank])
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    optimize_fn = optimization_manager(global_rank, config)
    eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    ckpt_path = os.path.join(checkpoint_dir, config.ckpt_file)
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Evaluating at training step %d' % step)
    dist.barrier()

    if config.eval_loss:
        eval_loss = compute_eval_loss(
            config, sde, state, eval_step_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Testing loss: %.5f" % eval_loss.item())
        dist.barrier()

    if config.eval_likelihood:
        mean_nll = compute_non_image_likelihood(
            config, sde, state, likelihood_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Mean NLL: %.5f" % mean_nll.item())
        dist.barrier()

    if config.eval_sample:
        x, _, nfe = sampling_fn(score_model)
        logging.info('NFE: %d' % nfe)

        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], s=3)
        plt.savefig(os.path.join(sample_dir, 'sample_%d.png' % global_rank))
        plt.close()
