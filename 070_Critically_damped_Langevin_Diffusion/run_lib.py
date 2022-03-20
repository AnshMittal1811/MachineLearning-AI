# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import glob
import logging
import time
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from util.utils import calculate_frechet_distance
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from models import ncsnpp
from util.utils import make_dir, get_optimizer, optimization_manager, get_data_scaler, get_data_inverse_scaler, set_seeds, save_img
from util.utils import compute_eval_loss, compute_image_likelihood, broadcast_params, reduce_tensor, build_beta_fn, build_beta_int_fn
from util import datasets
from util.checkpoint import save_checkpoint, restore_checkpoint
import losses
import sde_lib
import sampling
import likelihood
import evaluation


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
    fid_dir = os.path.join(workdir, 'fid')

    if global_rank == 0:
        logging.info(config)
        if config.mode == 'train':
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
            make_dir(likelihood_dir)
            make_dir(fid_dir)
            make_dir(tb_dir)
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
    broadcast_params(score_model.parameters())
    score_model = DDP(score_model, device_ids=[local_rank])

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    if global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, score_model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    dist.barrier()

    # Utility functions to map images from [0, 1] to [-1, 1] and back.
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

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

    train_queue, valid_queue, _ = datasets.get_loaders(config)
    num_total_iter = config.n_train_iters

    if global_rank == 0:
        logging.info('Number of total iterations: %d' % num_total_iter)
    dist.barrier()

    optimize_fn = optimization_manager(config)
    train_step_fn = losses.get_step_fn(True, optimize_fn, sde, config)
    eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.image_channels,
                      config.image_size,
                      config.image_size)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    inceptionv3 = config.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Starting training at step %d' % step)
    dist.barrier()

    if config.mode == 'continue':
        config.eval_threshold = max(step + 1, config.eval_threshold)
        config.snapshot_threshold = max(step + 1, config.snapshot_threshold)
        config.likelihood_threshold = max(
            step + 1, config.likelihood_threshold)
        config.fid_threshold = max(step + 1, config.fid_threshold)
        config.save_threshold = max(step + 1, config.save_threshold)

    while step < num_total_iter:
        for _, (train_x, _) in enumerate(train_queue):
            if step >= num_total_iter:
                break

            if step % config.eval_freq == 0 and step >= config.eval_threshold:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                eval_loss = compute_eval_loss(
                    config, state, eval_step_fn, valid_queue, scaler)
                ema.restore(score_model.parameters())

                if global_rank == 0:
                    logging.info('Testing loss at step: %d: %.5f' %
                                 (step, eval_loss.item()))
                    writer.add_scalar('eval_loss', eval_loss.item(), step)
                dist.barrier()

            if step % config.likelihood_freq == 0 and step >= config.likelihood_threshold:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                mean_nll = compute_image_likelihood(
                    config, sde, state, likelihood_fn, scaler, inverse_scaler, valid_queue, step=step, likelihood_dir=likelihood_dir)
                ema.restore(score_model.parameters())

                if global_rank == 0:
                    logging.info('Mean NLL (in BPD) at step: %d: %.5f' %
                                 (step, mean_nll.item()))
                    writer.add_scalar('mean_nll', mean_nll.item(), step)

                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % step)
                    if not os.path.isfile(checkpoint_file):
                        save_checkpoint(checkpoint_file, state)
                dist.barrier()

            if step % config.snapshot_freq == 0 and global_rank == 0 and step >= config.snapshot_threshold:
                logging.info('Saving snapshot checkpoint.')
                save_checkpoint(os.path.join(
                    checkpoint_dir, 'checkpoint.pth'), state)

                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                x, v, nfe = sampling_fn(score_model)
                ema.restore(score_model.parameters())

                x = inverse_scaler(x)
                logging.info('NFE for snapshot at step %d: %d' % (step, nfe))
                writer.add_scalar('nfe', nfe, step)

                this_sample_dir = os.path.join(sample_dir, 'iter_%d' % step)
                make_dir(this_sample_dir)
                save_img(x.clamp(0.0, 1.0), os.path.join(
                    this_sample_dir, 'sample.png'))

                if config.sde == 'uld':
                    np.save(os.path.join(this_sample_dir, 'sample_x'), x.cpu())
                    np.save(os.path.join(this_sample_dir, 'sample_v'), v.cpu())
                else:
                    np.save(os.path.join(this_sample_dir, 'sample'), x.cpu())
            dist.barrier()

            if step % config.fid_freq == 0 and step >= config.fid_threshold:
                this_sample_dir = os.path.join(fid_dir, 'step_%d' % step)
                if global_rank == 0:
                    make_dir(this_sample_dir)
                dist.barrier()

                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())

                num_sampling_rounds = config.eval_fid_samples // (
                    config.sampling_batch_size * global_size) + 1

                for r in range(num_sampling_rounds):
                    if global_rank == 0:
                        logging.info('sampling -- round: %d' % r)
                    dist.barrier()

                    x, _, nfe = sampling_fn(score_model)
                    x = inverse_scaler(x)

                    samples = np.clip(x.permute(0, 2, 3, 1).cpu(
                    ).numpy() * 255., 0, 255).astype(np.uint8)
                    samples = samples.reshape(
                        (-1, config.image_size, config.image_size, config.image_channels))

                    latents = evaluation.run_inception_distributed(
                        samples, inception_model, inceptionv3=inceptionv3)
                    np.save(os.path.join(fid_dir, 'statistics_%d_rank_%d_pool.npy' % (
                        r, global_rank)), latents['pool_3'])
                    np.save(os.path.join(fid_dir, 'nfes_%d_%d.npy' %
                            (r, global_rank)), np.array([nfe]))
                    np.save(os.path.join(fid_dir, 'samples_%d_%d.npy' %
                            (r, global_rank)), samples)

                dist.barrier()
                ema.restore(score_model.parameters())

                all_pool = []
                for pool_file in glob.glob(os.path.join(fid_dir, 'statistics_*_pool.npy')):
                    stat = np.load(pool_file)
                    all_pool.append(stat)
                all_pool = np.concatenate(all_pool, axis=0)[
                    :config.eval_fid_samples]
                if all_pool.shape[0] != config.eval_fid_samples:
                    raise ValueError('Not enough FID samples.')

                all_nfes = []
                for nfe_file in glob.glob(os.path.join(fid_dir, 'nfes_*.npy')):
                    nfe = np.load(nfe_file)
                    all_nfes.append(nfe)
                all_nfes = np.concatenate(all_nfes, axis=0)

                if global_rank == 0:
                    data_stats = evaluation.load_dataset_stats(config)
                    data_pools = data_stats['pool_3']
                    data_pools_mean = np.mean(data_pools, axis=0)
                    data_pools_sigma = np.cov(data_pools, rowvar=False)
                    all_pool_mean = np.mean(all_pool, axis=0)
                    all_pool_sigma = np.cov(all_pool, rowvar=False)

                    fid = calculate_frechet_distance(
                        data_pools_mean, data_pools_sigma, all_pool_mean, all_pool_sigma)
                    logging.info('FID: %.6f' % fid)
                    result_arr = np.array([fid])
                    np.save(os.path.join(fid_dir, 'report.npy'), result_arr)

                    mean_nfe = np.mean(all_nfes)
                    logging.info('Mean NFE: %.3f' % mean_nfe)

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

            x = scaler(train_x)
            x = x.to(config.device)
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
    global_size = config.global_size
    set_seeds(global_rank, config.seed + config.eval_seed)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    eval_dir = os.path.join(workdir, config.eval_folder)
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    fid_dir = os.path.join(eval_dir, 'fid')
    samples_dir = os.path.join(eval_dir, 'samples')
    if global_rank == 0:
        logging.info(config)
        make_dir(fid_dir)
        make_dir(samples_dir)
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

    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    _, valid_queue, _ = datasets.get_loaders(config)

    optimize_fn = optimization_manager(config)
    eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.image_channels,
                      config.image_size,
                      config.image_size)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    inceptionv3 = config.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    ckpt_path = os.path.join(checkpoint_dir, config.ckpt_file)
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Evaluating at training step %d' % step)
    dist.barrier()

    if config.eval_loss:
        eval_loss = compute_eval_loss(
            config, state, eval_step_fn, valid_queue, scaler, test=True)
        if global_rank == 0:
            logging.info('Testing loss: %.5f' % eval_loss.item())
        dist.barrier()

    if config.eval_likelihood:
        mean_bpd = compute_image_likelihood(
            config, sde, state, likelihood_fn, scaler, inverse_scaler, valid_queue, test=True)
        if global_rank == 0:
            logging.info('Mean NLL: %.5f' % mean_bpd.item())
        dist.barrier()

    if config.eval_fid:
        num_sampling_rounds = config.eval_fid_samples // (
            config.sampling_batch_size * global_size) + 1

        for r in range(num_sampling_rounds):
            if global_rank == 0:
                logging.info('sampling -- round: %d' % r)
            dist.barrier()

            x, _, nfe = sampling_fn(score_model)
            x = inverse_scaler(x)

            samples = np.clip(x.permute(0, 2, 3, 1).cpu().numpy()
                              * 255., 0, 255).astype(np.uint8)
            samples = samples.reshape(
                (-1, config.image_size, config.image_size, config.image_channels))

            latents = evaluation.run_inception_distributed(
                samples, inception_model, inceptionv3=inceptionv3)
            np.save(os.path.join(fid_dir, 'statistics_%d_rank_%d_pool.npy' %
                    (r, global_rank)), latents['pool_3'])
            np.save(os.path.join(fid_dir, 'nfes_%d_%d.npy' %
                    (r, global_rank)), np.array([nfe]))
            np.save(os.path.join(fid_dir, 'samples_%d_%d.npy' %
                    (r, global_rank)), samples)

        dist.barrier()

        all_pool = []
        for pool_file in glob.glob(os.path.join(fid_dir, 'statistics_*_pool.npy')):
            stat = np.load(pool_file)
            all_pool.append(stat)
        all_pool = np.concatenate(all_pool, axis=0)[:config.eval_fid_samples]
        if all_pool.shape[0] != config.eval_fid_samples:
            raise ValueError('Not enough FID samples.')

        all_nfes = []
        for nfe_file in glob.glob(os.path.join(fid_dir, 'nfes_*.npy')):
            nfe = np.load(nfe_file)
            all_nfes.append(nfe)
        all_nfes = np.concatenate(all_nfes, axis=0)

        if global_rank == 0:
            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats['pool_3']
            data_pools_mean = np.mean(data_pools, axis=0)
            data_pools_sigma = np.cov(data_pools, rowvar=False)
            all_pool_mean = np.mean(all_pool, axis=0)
            all_pool_sigma = np.cov(all_pool, rowvar=False)

            fid = calculate_frechet_distance(
                data_pools_mean, data_pools_sigma, all_pool_mean, all_pool_sigma)
            logging.info('FID: %.6f' % fid)
            result_arr = np.array([fid])
            np.save(os.path.join(fid_dir, 'report.npy'), result_arr)

            mean_nfe = np.mean(all_nfes)
            logging.info('Mean NFE: %.3f' % mean_nfe)

        dist.barrier()

    if config.eval_sample:
        num_sampling_rounds = config.eval_sample_samples // (
            config.sampling_batch_size * global_size) + 1

        for r in range(num_sampling_rounds):
            if global_rank == 0:
                logging.info('sampling -- round: %d' % r)
            dist.barrier()

            x, _, nfe = sampling_fn(score_model)
            x = inverse_scaler(x)
            samples = x.clamp(0.0, 1.0)

            torch.save(samples, os.path.join(
                samples_dir, 'samples_%d_%d.pth' % (r, global_rank)))
            np.save(os.path.join(samples_dir, 'nfes_%d_%d.npy' %
                    (r, global_rank)), np.array([nfe]))

        dist.barrier()

        if global_rank == 0:
            all_samples = []
            for sample_file in glob.glob(os.path.join(samples_dir, 'samples_*.pth')):
                sample = torch.load(sample_file, map_location=config.device)
                all_samples.append(sample)

            all_samples = torch.cat(all_samples)
            torch.save(all_samples, os.path.join(
                samples_dir, 'all_samples.pth'))

            all_nfes = []
            for nfe_file in glob.glob(os.path.join(samples_dir, 'nfes_*.npy')):
                nfe = np.load(nfe_file)
                all_nfes.append(nfe)
            all_nfes = np.concatenate(all_nfes, axis=0)

            mean_nfe = np.mean(all_nfes)
            logging.info('Mean NFE: %.3f' % mean_nfe)
        dist.barrier()
