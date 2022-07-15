"""
Training process.
$ python train.py
"""

import os, sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from misc import options
from misc import utils
from misc import metric


cudnn.benchmark = True
cudnn.deterministic = True


def main():
    # Setup workspace and backup files
    cfg = options.get_config()
    workspace = utils.setup_workspace(cfg.workspace)
    if cfg.pretrained is not None:
        logger = utils.Logger(os.path.join(workspace.log, 'train_log.txt'), mode='a')
    else:
        logger = utils.Logger(os.path.join(workspace.log, 'train_log.txt'))
    tf_logger = SummaryWriter(workspace.log)
    logger.write('Workspace: {}'.format(cfg.workspace), 'green')
    logger.write('CUDA: {}, Multi-GPU: {}'.format(cfg.cuda, cfg.multi_gpu), 'green')
    logger.write('To-disparity: {}'.format(cfg.to_disparity), 'green')

    # Define dataloader
    logger.write('Dataset: {}'.format(cfg.dataset_name), 'green')
    train_dataset, val_dataset = options.get_dataset(cfg.dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, sampler=None,
                              worker_init_fn=lambda work_id: np.random.seed(work_id))
                              # worker_init_fn ensures different sampling patterns for
                              # each data loading thread
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                            num_workers=cfg.workers)

    # Define model
    logger.write('Model: {}'.format(cfg.model_name), 'green')
    model = options.get_model(cfg.model_name)
    if cfg.multi_gpu:
        model = nn.DataParallel(model)
    if cfg.cuda:
        model = model.cuda()

    # Define loss function
    criterion = options.get_criterion(cfg.criterion_name)
    if cfg.cuda:
        criterion = criterion.cuda()
    logger.write('Criterion: {}'.format(criterion), 'green')

    # Define optimizer and learning rate scheduler
    optim = options.get_optimizer(cfg.optimizer_name, model.parameters())
    lr_scheduler = options.get_lr_scheduler(cfg.lr_scheduler_name, optim)
    logger.write('Optimizer: {}'.format(optim), 'green')
    if lr_scheduler is not None:
        logger.write('Learning rate schedular: {}'.format(lr_scheduler), 'green')

    # [Optional] load pretrained model
    start_ep = 0
    global_step = 0
    local_start = 0
    if cfg.pretrained is not None:
        start_ep, global_step = utils.load_checkpoint(model, optim, lr_scheduler, cfg.pretrained, cfg.weight_only)
        logger.write('Load pretrained model from {}'.format(cfg.pretrained), 'green')
        #global_step = len(train_dataset) * start_ep # NOTE: global step start from the beginning of the epoch
        local_start = global_step % len(train_dataset)

    # Start training
    logger.write('Start training...', 'green')
    for ep in range(start_ep, cfg.max_epoch):
        if lr_scheduler is not None:
            logger.write('Update learning rate: {} --> '.format(lr_scheduler.get_lr()[0]), 'magenta', end='')
            lr_scheduler.step()
            logger.write('{}'.format(lr_scheduler.get_lr()[0]), 'magenta')

        # Train an epoch
        model.train()
        meters = metric.Metrics(cfg.train_metric_field)
        avg_meters = metric.MovingAverageEstimator(cfg.train_metric_field)
        end = time.time()
        for it, data in enumerate(train_loader, local_start):
            # Pack data
            if cfg.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            if cfg.to_disparity:
                inputs['left_sd'] = data['left_sdisp']
                inputs['right_sd'] = data['right_sdisp']
                target = data['left_disp']
            else:
                inputs['left_sd'] = data['left_sd']
                inputs['right_sd'] = data['right_sd']
                target = data['left_d']
            data_time = time.time() - end

            # Inference, compute loss and update model
            end = time.time()
            optim.zero_grad()
            pred = model(inputs)
            if cfg.criterion_name in ['inv_disp_l1']:
                pred_d = utils.disp2depth(pred, data['width'].item())
                loss = criterion(pred_d, data['left_d'])
            else:
                loss = criterion(pred, target)
            loss.backward()
            optim.step()
            update_time = time.time() - end
            end = time.time()

            # Measure performance
            pred_np = pred.data.cpu().numpy()
            target_np = target.data.cpu().numpy()
            results = meters.compute(pred_np, target_np)
            avg_meters.update(results)

            # Print results
            if (it % cfg.print_step) == 0:
                logger.write('[{:2d}/{:2d}][{:5d}/{:5d}] data time: {:4.3f}, update time: {:4.3f}, loss: {:.4f}'\
                             .format(ep, cfg.max_epoch, it, len(train_loader), data_time,
                                     update_time, loss.item()))
                avg_results = avg_meters.compute()
                logger.write('   [Average results] ', end='')
                for key, val in avg_results.items():
                    logger.write('{}: {:5.3f} '.format(key, val), end='')
                logger.write('')
                avg_meters.reset()

            # Log to tensorboard
            if (it % cfg.tflog_step) == 0:
                tf_logger.add_scalar('A-Loss/loss', loss.data, global_step)
                for key, val in results.items():
                    tf_logger.add_scalar('B-Train-Dense-Metric/{}'.format(key), val, global_step)
                if cfg.lr_scheduler_name is not None:
                    tf_logger.add_scalar('C-Learning-Rate', lr_scheduler.get_lr()[0], global_step)
                tf_logger.add_image('A-RGB/left', inputs['left_rgb'].data, global_step)
                tf_logger.add_image('A-RGB/right', inputs['right_rgb'].data, global_step)
                norm_factor = target.data.max(-1)[0].max(-1)[0].max(-1)[0][:, None, None, None]
                tf_logger.add_image('B-sD', inputs['left_sd'].data / norm_factor, global_step)
                tf_logger.add_image('C-Pred', pred.data / norm_factor, global_step)
                tf_logger.add_image('C-Ground-Truth', target.data / norm_factor, global_step)
                if cfg.dump_all_param: # NOTE: this will require a lot of HDD memory
                    for name, param in model.named_parameters():
                        tf_logger.add_histogram(name+'/vars', param.data.clone().cpu().numpy(), global_step)
                        if param.requires_grad:
                            tf_logger.add_histogram(name+'/grads', param.grad.clone().cpu().numpy(), global_step)

            # On-the-fly validation
            if (it % cfg.val_step) == 0:# and not (ep == 0 and it == 0):
                validate(global_step, val_loader, model, logger, tf_logger, cfg)

            # Save model
            if (it % cfg.save_step) == 0:
                ckpt_path = utils.save_checkpoint(workspace.ckpt, model, optim, lr_scheduler, ep, global_step)
                logger.write('Save checkpoint to {}'.format(ckpt_path), 'magenta')

            # Update global step
            global_step += 1

            if it >= len(train_dataset):
                local_start = 0
                break


def validate(global_step, loader, model, logger, tf_logger, cfg):
    model.eval()

    pbar = tqdm(loader)
    pbar.set_description('Online validation')
    disp_meters = metric.Metrics(['err_3px'])
    disp_avg_meters = metric.MovingAverageEstimator(['err_3px'])
    depth_meters = metric.Metrics(cfg.val_metric_field)
    depth_avg_meters = metric.MovingAverageEstimator(cfg.val_metric_field)
    with torch.no_grad():
        for it, data in enumerate(pbar):
            # Pack data
            if cfg.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            if cfg.to_disparity:
                inputs['left_sd'] = data['left_sdisp']
                inputs['right_sd'] = data['right_sdisp']
            else:
                inputs['left_sd'] = data['left_sd']
                inputs['right_sd'] = data['right_sd']
            target_d = data['left_d']
            target_disp = data['left_disp']
            img_w = data['width'].item()

            # Inference
            pred = model(inputs)
            if cfg.to_disparity:
                pred_d = utils.disp2depth(pred, img_w)
                pred_disp = pred
            else:
                raise NotImplementedError

            # Measure performance
            if cfg.to_disparity:
                # disparity
                pred_disp_np = pred_disp.data.cpu().numpy()
                target_disp_np = target_disp.data.cpu().numpy()
                disp_results = disp_meters.compute(pred_disp_np, target_disp_np)
                disp_avg_meters.update(disp_results)
                # depth
                pred_d_np = pred_d.data.cpu().numpy()
                target_d_np = target_d.data.cpu().numpy()
                depth_results = depth_meters.compute(pred_d_np, target_d_np)
                depth_avg_meters.update(depth_results)
            else:
                raise NotImplementedError

    logger.write('\nValidation results: ', 'magenta')
    if cfg.to_disparity:
        disp_avg_results = disp_avg_meters.compute()
        for key, val in disp_avg_results.items():
            logger.write('- [disparity] {}: {}'.format(key, val), 'magenta')
            tf_logger.add_scalar('B-Val-Dense-Metric/disp-{}'.format(key), val, global_step)
    depth_avg_results = depth_avg_meters.compute()
    for key, val in depth_avg_results.items():
        logger.write('- [depth] {}: {}'.format(key, val), 'magenta')
        tf_logger.add_scalar('B-Val-Dense-Metric/depth-{}'.format(key), val, global_step)
    logger.write('\n')
    
    # NOTE: remember to set back to train mode after on-the-fly validation
    model.train()


if __name__ == '__main__':
    main()
