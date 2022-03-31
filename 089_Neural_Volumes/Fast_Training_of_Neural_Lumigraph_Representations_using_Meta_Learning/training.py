'''Implements a generic training loop.
'''
from pathlib import Path
import time
import os

import torch
from torch import nn
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import numpy as np
from modules_sdf import SDFIBRNet
from torchmeta.modules.utils import get_subdict

import utils.common_utils as common_utils
import utils.diff_operators as diff_operators
from scheduler import Scheduler


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, clip_grad=False, loss_schedules=None, optim=None, verbose_record_file: Path = None,
          teacher: SDFIBRNet = None, meta_spec_log=False, ibr_log=False):

    no_optimization = 0
    if optim is None:
        # Dummy optimizer.
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
        no_optimization = 1

    os.makedirs(model_dir, 0o777, True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    common_utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    common_utils.cond_mkdir(checkpoints_dir)

    verbose_dir = None
    loss_log = []
    if model.opt.verbose_logging:
        verbose_dir = Path(model_dir) / 'verbose'
        verbose_dir.mkdir(0o777, True, True)

    writer = SummaryWriter(summaries_dir)
    device = model.device

    # Scheduler.
    scheduler = Scheduler(model.opt, model, optim, writer)

    # Restore verbose record.
    epoch_0 = 0
    total_steps = 0
    verbose_record = None
    if verbose_record_file is not None:
        verbose_record = torch.load(verbose_record_file, map_location=model.device)

        optim.load_state_dict(verbose_record['optimizer_state_dict'])
        model.load_state_dict(verbose_record['model_state_dict'], strict=False)
        scheduler.update(verbose_record['epoch'], verbose_record['step'] - 1, verbose_record['loss_log'][-2])

        epoch_0 = verbose_record['epoch']
        total_steps = verbose_record['step']

    print(f'Will train for {epochs} epochs with {len(train_dataloader)} iterations per epoch.')

    train_loss = torch.Tensor([0])
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epoch_0, epochs):
            if is_event_step_log(epoch, epochs_til_checkpoint, orig_log=True):
                # Regular checkpoint
                save_checkpoint(checkpoints_dir, model, optim, epoch, total_steps, train_loss.item(), True, False, orig_log=True)

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                # Override inputs for verbose record iter 0
                if verbose_record is not None and total_steps == verbose_record['step']:
                    model_input = verbose_record['inputs']
                    gt = verbose_record['gt']

                # To GPU.
                for k, v in model_input.items():
                    v = v.to(device)
                    if v.dtype == torch.float:
                        v = v.requires_grad_(True)
                    model_input[k] = v

                gt = {key: value.to(device) for key, value in gt.items()}

                # If teacher exists, override GT.
                if teacher is not None:
                    gt = query_teacher(teacher, model_input, gt)

                # Renormalize params.
                model.renormalize()

                if ibr_log:
                    if total_steps <= 5000:
                        model.opt.occ_threshold = 1e-3
                    elif 5000 < total_steps <= 10000:
                        model.opt.occ_threshold = 1e-4
                    else:
                        model.opt.occ_threshold = 1e-5
                elif meta_spec_log:
                    if total_steps <= 3000:
                        model.opt.occ_threshold = 1e-3
                    elif 3000 < total_steps <= 8000:
                        model.opt.occ_threshold = 1e-4
                    else:
                        model.opt.occ_threshold = 1e-5

                if model.opt.dataset_name == 'shapenet':
                    OPTSHAPE_FIRST = 2000
                    OPTSHAPE_EVERY = 5
                elif model.opt.dataset_name == 'dtu':
                    OPTSHAPE_FIRST = 50
                    OPTSHAPE_EVERY = 7
                elif model.opt.dataset_name == 'nlr':
                    OPTSHAPE_FIRST = 100
                    OPTSHAPE_EVERY = 3

                # Forward.
                t1iter = time.time()
                if total_steps < OPTSHAPE_FIRST:
                    model_input['train_shape'] = 1
                elif total_steps % OPTSHAPE_EVERY == 0:
                    model_input['train_shape'] = 1
                else:
                    model_input['train_shape'] = 0
                    scheduler.set_lr('sdf', 0, total_steps)

                model_output = model(model_input)

                model_output['weights_sdf'] = {k: v for k, v in model.decoder_sdf.named_parameters()}
                liter = time.time() - t1iter
                # print(f'Iteration Length: {liter}')

                t1loss = time.time()
                # Compute losses.
                losses = loss_fn(model_output, gt)
                lloss = time.time() - t1loss
                # print(f'Loss Length {lloss}')

                # Sum only active losses for optimization.
                train_loss = 0.
                for loss_name, (loss, loss_enabled) in losses.items():
                    single_loss = loss.mean()
                    if torch.isnan(single_loss).any().item():
                        print('We have NAN in loss!!!!')
                        import pdb
                        pdb.set_trace()
                        raise Exception('NaN in loss!')

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # Optionally apply schedule.
                        schedule = loss_schedules[loss_name](total_steps)
                        writer.add_scalar(loss_name + "_weight", schedule, total_steps)
                        single_loss *= schedule

                    writer.add_scalar(loss_name, single_loss, total_steps)

                    if loss_enabled:
                        # Sum only active losses.
                        train_loss += single_loss

                # Log summary loss.
                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                writer.add_scalar("swish_beta", np.mean(
                    [m.beta.mean().item() for m in model.modules() if type(m).__name__ in ['Swish']]), total_steps)

                # Verify rerun
                if verbose_record is not None and total_steps == verbose_record['step']:
                    assert abs(train_loss.item() - verbose_record['loss'].item()) < 1e-5

                # Summarize?
                if is_event_step_log(total_steps, steps_til_summary, meta_spec_log=meta_spec_log, ibr_log=ibr_log,
                                     orig_log=(not (meta_spec_log or ibr_log))):
                    # Current checkpoint.
                    save_checkpoint(checkpoints_dir, model, optim, epoch, total_steps, train_loss.item(), False, False, meta_spec_log=meta_spec_log,
                                    ibr_log=ibr_log, orig_log=(not (meta_spec_log or ibr_log)))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)
                elif verbose_record:
                    # Render
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)
                    pass

                # Detailed verbose logging.
                if model.opt.verbose_logging:
                    loss_log += [train_loss.item()]
                    torch.save({
                        'epoch': epoch,
                        'step': total_steps,
                        'loss': train_loss,
                        'losses': losses,
                        'loss_log': torch.from_numpy(np.array(loss_log)),
                        'optimizer_state_dict': optim.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'git_revision': common_utils.get_git_revision(),
                        'inputs': model_input,
                        'gt': gt,
                    }, verbose_dir / f'{total_steps:08d}.pth')

                # Backward.
                t1backward = time.time()
                optim.zero_grad()
                if not no_optimization:
                    train_loss.backward()
                lbackward = time.time() - t1backward
                # print(f'Backward Length: {lbackward}')

                grads_isnan = {k: torch.isnan(x.grad).any().item()
                               for k, x in model.named_parameters() if x.grad is not None}
                if np.any(list(grads_isnan.values())):
                    print('We have NAN in gradients!!!!')
                    import pdb
                    pdb.set_trace()
                    torch.save(model.state_dict(), Path(checkpoints_dir) / 'model_broken.pth')
                    torch.save(model_input, Path(checkpoints_dir) / 'model_inputs.pth')
                    torch.save(gt, Path(checkpoints_dir) / 'model_gt.pth')
                    raise Exception('NaN in gradients!')

                t1optim = time.time()
                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()
                loptim = time.time() - t1optim
                # print(f'Optimizer Length: {loptim}')

                total_time = liter + lloss + lbackward + loptim
                writer.add_scalar("iteration_total_time", total_time, total_steps)

                if not no_optimization:
                    scheduler.update(epoch, total_steps, train_loss)

                params_isnan = [torch.isnan(x).any().item() for x in model.parameters()]
                if np.any(params_isnan):
                    print('We have NAN in parameters!!!!')
                    import pdb
                    pdb.set_trace()
                    raise Exception('NaN in parameters!')

                pbar.update(1)

                if is_event_step_log(total_steps, steps_til_summary, meta_spec_log=meta_spec_log, ibr_log=ibr_log,
                                     orig_log=(not (meta_spec_log or ibr_log))):
                    tqdm.write("Epoch %d/%d, Total loss %0.6f, iteration time %0.6f" %
                               (epoch, epochs, train_loss.item(), time.time() - start_time))
                total_steps += 1

        # Final checkpoint.
        save_checkpoint(checkpoints_dir, model, optim, epoch, total_steps, train_loss.item(), True, True, orig_log=True)


def train_meta(meta_model, meta_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, clip_grad=False, loss_schedules=None, optim=None, verbose_record_file: Path = None,
          teacher: SDFIBRNet = None):

    if optim is None:
        optim = torch.optim.Adam(lr=lr, params=meta_model.parameters())

    os.makedirs(model_dir, 0o777, True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    common_utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    common_utils.cond_mkdir(checkpoints_dir)

    verbose_dir = None
    loss_log = []
    if meta_model.opt.verbose_logging:
        verbose_dir = Path(model_dir) / 'verbose'
        verbose_dir.mkdir(0o777, True, True)

    writer = SummaryWriter(summaries_dir)
    device = meta_model.device

    # Scheduler.
    scheduler = Scheduler(meta_model.opt, meta_model, optim, writer)

    # Restore verbose record.
    epoch_0 = 0
    total_steps = 0
    verbose_record = None
    if verbose_record_file is not None:
        verbose_record = torch.load(verbose_record_file, map_location=meta_model.device)

        optim.load_state_dict(verbose_record['optimizer_state_dict'])
        meta_model.load_state_dict(verbose_record['model_state_dict'], strict=False)
        scheduler.update(verbose_record['epoch'], verbose_record['step'] - 1, verbose_record['loss_log'][-2])

        epoch_0 = verbose_record['epoch']
        total_steps = verbose_record['step']

    print(f'Will train for {epochs} epochs with {len(meta_dataloader)} iterations per epoch.')

    meta_loss = torch.Tensor([0])
    with tqdm(total=len(meta_dataloader) * epochs) as pbar:
        meta_losses = []
        for epoch in range(epoch_0, epochs):
            if is_event_step_log(epoch, epochs_til_checkpoint, orig_log=True):
                # Regular checkpoint
                save_checkpoint(checkpoints_dir, meta_model, optim, epoch, total_steps, meta_loss.item(), True, False, orig_log=True)

            for step, meta_batch in enumerate(meta_dataloader):
                start_time = time.time()

                # Override inputs for verbose record iter 0
                if verbose_record is not None and total_steps == verbose_record['step']:
                    meta_batch = verbose_record['meta_batch']

                # To GPU.
                def dict_to_gpu(d):
                    for k, v in d.items():
                        v = v.to(device)
                        if v.dtype == torch.float:
                            v = v.requires_grad_(True)
                        d[k] = v

                    return d

                for k, v in meta_batch['context'].items():
                    v = [dict_to_gpu(item) for item in v]
                    meta_batch['context'][k] = v

                for k, v in meta_batch['query'].items():
                    v = dict_to_gpu(v)
                    meta_batch['query'][k] = v

                # Renormalize params.
                meta_model.renormalize()

                # Forward.
                meta_model_output = meta_model(meta_batch)

                # Add model parameters for regularization losses.
                meta_model_output['model_out']['weights_sdf'] = {k: v for k, v in get_subdict(meta_model_output['fast_params'], 'decoder_sdf').items()}

                # Currently meta_loss is a placeholder, since optimization is done via REPTILE for weight updates
                meta_loss = torch.tensor([0])

                # Log summary loss.
                meta_losses.append(meta_loss.item())
                writer.add_scalar("total_train_loss", meta_loss, total_steps)

                # Verify rerun
                if verbose_record is not None and total_steps == verbose_record['step']:
                    assert abs(meta_loss.item() - verbose_record['loss'].item()) < 1e-5

                # Summarize?
                if is_event_step_log(total_steps, steps_til_summary):
                    # Current checkpoint.
                    save_checkpoint(checkpoints_dir, meta_model, optim, epoch, total_steps, meta_loss.item(), False, False, meta_learning_log=True)
                    summary_fn(meta_model, meta_batch, meta_model_output, writer, total_steps)
                elif verbose_record:
                    # Render
                    summary_fn(meta_model, meta_batch, meta_model_output, writer, total_steps)
                    pass

                # Detailed verbose logging.
                if meta_model.opt.verbose_logging:
                    loss_log += [meta_loss.item()]
                    torch.save({
                        'epoch': epoch,
                        'step': total_steps,
                        'loss': meta_loss,
                        'loss_log': torch.from_numpy(np.array(loss_log)),
                        'optimizer_state_dict': optim.state_dict(),
                        'model_state_dict': meta_model.state_dict(),
                        'git_revision': common_utils.get_git_revision(),
                        'meta_batch': meta_batch,
                    }, verbose_dir / f'{total_steps:08d}.pth')

                # Backward.
                optim.zero_grad()

                if meta_model.opt.meta_algorithm == 'maml':
                    meta_loss.backward()

                    grads_isnan = {k: torch.isnan(x.grad).any().item()
                                   for k, x in meta_model.named_parameters() if x.grad is not None}
                    if np.any(list(grads_isnan.values())):
                        print('We have NAN in gradients!!!!')
                        import pdb
                        pdb.set_trace()
                        torch.save(meta_model.state_dict(), Path(checkpoints_dir) / 'model_broken.pth')
                        torch.save(meta_batch, Path(checkpoints_dir) / 'meta_batch.pth')
                        raise Exception('NaN in gradients!')

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=clip_grad)

                    optim.step()
                elif meta_model.opt.meta_algorithm == 'reptile':
                    meta_model._update_meta_params(meta_model_output['fast_params'], meta_model.opt.meta_lr)
                else:
                    print('Not a valid meta-learning algorithm!')
                    raise Exception('Not a valid meta-learning algorithm!')

                scheduler.update(epoch, total_steps, meta_loss)
                update_meta_lr(meta_model, total_steps)

                params_isnan = [torch.isnan(x).any().item() for x in meta_model.parameters()]
                if np.any(params_isnan):
                    print('We have NAN in parameters!!!!')
                    import pdb
                    pdb.set_trace()
                    raise Exception('NaN in parameters!')

                pbar.update(1)

                if is_event_step_log(total_steps, steps_til_summary):
                    tqdm.write("Epoch %d/%d, Total loss %0.6f, iteration time %0.6f" %
                               (epoch, epochs, meta_loss.item(), time.time() - start_time))
                total_steps += 1

        # Final checkpoint.
        save_checkpoint(checkpoints_dir, meta_model, optim, epoch, total_steps, meta_loss.item(), True, True, orig_log=True)


def save_checkpoint(checkpoints_dir, model, optim, epoch, steps, loss, is_regular=False, is_final=False, meta_spec_log=False, orig_log=False,
                    meta_learning_log=False, ibr_log=False):
    """
    Saves full training state.
    """
    suffix = 'final' if is_final else 'current'
    checkpoints_dir = Path(checkpoints_dir)
    # Model.
    torch.save(model.state_dict(), checkpoints_dir / f'model_{suffix}.pth')
    if is_regular and orig_log:
        torch.save(model.state_dict(), checkpoints_dir / f'model_epoch_{epoch:04d}.pth')
    if meta_spec_log or meta_learning_log or ibr_log or orig_log:
        torch.save(model.state_dict(), checkpoints_dir / f'model_iter_{steps:08d}.pth')

    # Optimizer.
    train_state = {
        'epoch': epoch,
        'step': steps,
        'loss': loss,
        'optimizer_state_dict': optim.state_dict(),
        'git_revision': common_utils.get_git_revision()
    }
    torch.save(train_state, checkpoints_dir / f'optim_{suffix}.pth')
    if is_regular and orig_log:
        torch.save(train_state, checkpoints_dir / f'optim_epoch_{epoch:04d}.pth')
    if meta_spec_log or meta_learning_log or ibr_log or orig_log:
        torch.save(train_state, checkpoints_dir / f'optim_iter_{steps:08d}.pth')


@torch.enable_grad()
def query_teacher(teacher: SDFIBRNet, model_input, gt):
    """
    Computes updated GT based on teacher.
    """
    input_pcd = {
        'coords': model_input['coords'],
        'time': model_input['time'],
    }
    teacher_out = teacher.decoder_sdf(input_pcd)
    sdf = teacher_out['model_out'][..., :1]
    normals = diff_operators.gradient(sdf, teacher_out['model_in'])[0, ...]

    # Override SDF ground-truth.
    gt['sdf'] = sdf.detach()
    gt['normals'] = normals[None, ...].detach()
    gt['is_sdf_explicit'] = True
    return gt


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


def is_event_step_log(step: int, event_freq_log: int = 1, meta_spec_log=False, orig_log=False, ibr_log=False):
    if meta_spec_log:
        # return is_event_step_log_meta_spec(step, event_freq_log)
        if step in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200]:
            return 1
        return step % 100 == 0
    if orig_log:
        # return is_event_step_log_orig(step, event_freq_log)
        if step in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 200]:
            return 1
        return step % 100 == 0
    if ibr_log:
        if step in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 200]:
            return 1
        return step % 100 == 0
    return step % 2 == 0
    # return True


def update_meta_lr(meta_model, total_steps: int):
    if (total_steps + 1) % 400 == 0:
        meta_model.update_lr(lr_sdf_new=meta_model.lr_sdf * 0.5)
    if (total_steps + 1) % 60 == 0:
        meta_model.update_lr(lr_new=meta_model.lr * 0.5)
    if (total_steps + 1) % 100000000 == 0:
        meta_model.update_lr(meta_lr=meta_model.opt.meta_lr * 0.5)
    if (total_steps + 1) % 100000000 == 0 and meta_model.num_meta_steps >= 32:
        meta_model.update_lr(meta_steps=int(meta_model.num_meta_steps/2))


def is_event_step_log_orig(step: int, event_freq_log: int = 1):
    """
    Checks if event should trigger using log rule.
    """
    step_log = step
    event_freq = event_freq_log
    while step_log // 10 > 0:
        step_log //= 10
        event_freq *= 10
        if event_freq >= 1e4:
            break  # Max period = 10000
    return step % event_freq == 0


def is_event_step_log_meta_spec(step: int, event_freq_log: int = 1):
    if step in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        return True
    if step == 1025:
        exit()

    return False
