import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
          loss_fn, summary_fn,
          prefix_model_dir='',
          val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          params=None):

    if params is None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=True)
    else:
        optim = torch.optim.Adam(lr=lr, params=params, amsgrad=True)

    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        pass
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)

    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                tmp = {}
                for key, value in model_input.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value.cuda()})
                    else:
                        tmp.update({key: value})
                model_input = tmp

                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    # summary_fn(model_input, gt, model_output, writer, total_steps)
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp


def train_wchunks(models, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
                  loss_fn, summary_fn, chunk_lists_from_batch_fn,
                  val_dataloader=None, double_precision=False, clip_grad=False, loss_schedules=None,
                  num_cuts=128,
                  weight_decay=0.0,
                  max_chunk_size=4096,
                  loss_start={},
                  resume_checkpoint={}):

    optims = {key: torch.optim.Adam(lr=lr, params=model.parameters())
              for key, model in models.items()}
    schedulers = {key: torch.optim.lr_scheduler.StepLR(optim, step_size=8000, gamma=0.2)
                  for key, optim in optims.items()}

    # load optimizer if supplied
    for key in models.keys():
        if key in resume_checkpoint:
            optims[key].load_state_dict(resume_checkpoint[key])
            schedulers = {key: torch.optim.lr_scheduler.StepLR(optim, step_size=8000, gamma=0.2)
                          for key, optim in optims.items()}

    if os.path.exists(os.path.join(model_dir, 'summaries')):
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    if 'total_steps' in resume_checkpoint:
        total_steps = resume_checkpoint['total_steps']

    start_epoch = 0
    if 'epoch' in resume_checkpoint:
        start_epoch = resume_checkpoint['epoch']
        for scheduler in schedulers.values():
            for i in range(start_epoch):
                scheduler.step()

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        pbar.update(total_steps)
        train_losses = []
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                for key, model in models.items():
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_'+key+'_epoch_%04d.pth' % epoch))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                               np.array(train_losses))
                for key, optim in optims.items():
                    torch.save({'epoch': epoch,
                                'total_steps': total_steps,
                                'optimizer_state_dict': optim.state_dict()},
                               os.path.join(checkpoints_dir, 'optim_'+key+'_epoch_%04d.pth' % epoch))

            for step, (model_input, meta, gt, misc) in enumerate(train_dataloader):
                start_time = time.time()

                for optim in optims.values():
                    optim.zero_grad()

                list_chunked_model_input, list_chunked_meta, list_chunked_gt = \
                    chunk_lists_from_batch_fn(model_input, meta, gt, max_chunk_size)

                num_chunks = len(list_chunked_gt)
                batch_avged_losses = {}
                batch_avged_tot_loss = 0.
                for chunk_idx, (chunked_model_input, chunked_meta, chunked_gt) \
                        in enumerate(zip(list_chunked_model_input, list_chunked_meta, list_chunked_gt)):
                    chunked_model_input = dict2cuda(chunked_model_input)
                    chunked_meta = dict2cuda(chunked_meta)
                    chunked_gt = dict2cuda(chunked_gt)

                    # forward pass through model
                    chunk_model_outputs = {key: model(chunked_model_input) for key, model in models.items()}
                    losses = loss_fn(chunk_model_outputs, chunked_gt,
                                     dataloader=train_dataloader)

                    # loss from forward pass
                    train_loss = 0.
                    for loss_name, loss in losses.items():

                        # slowly apply loss if less than start iter
                        if loss_name in loss_start:
                            if total_steps < loss_start[loss_name]:
                                loss = (total_steps / loss_start[loss_name])**2 * loss

                        single_loss = loss.mean()
                        train_loss += single_loss / num_chunks

                        batch_avged_tot_loss += float(single_loss / num_chunks)
                        if loss_name in batch_avged_losses:
                            batch_avged_losses[loss_name] += single_loss / num_chunks
                        else:
                            batch_avged_losses.update({loss_name: single_loss/num_chunks})

                    if weight_decay > 0:
                        for model in models.values():
                            train_loss += weight_decay * weight_decay_loss(model)
                    train_loss.backward()

                for loss_name, loss in batch_avged_losses.items():
                    writer.add_scalar(loss_name, loss, total_steps)
                train_losses.append(batch_avged_tot_loss)
                writer.add_scalar("total_train_loss", batch_avged_tot_loss, total_steps)

                if clip_grad:
                    for model in models.values():
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optims.values():
                    optim.step()

                if not total_steps % steps_til_summary:
                    for key, model in models.items():
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_'+key+'_current.pth'))
                    for key, optim in optims.items():
                        torch.save({'epoch': epoch,
                                    'total_steps': total_steps,
                                    'optimizer_state_dict': optim.state_dict()},
                                   os.path.join(checkpoints_dir, 'optim_'+key+'_current.pth'))
                    summary_fn(models, train_dataloader, val_dataloader, loss_fn, optims, meta, gt, misc,
                               writer, total_steps)

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1

            for scheduler in schedulers.values():
                scheduler.step()

        for key, model in models.items():
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_' + key + '_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def weight_decay_loss(model):
    L1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        elif 'weight' in name:
            L1_reg = L1_reg + torch.sum(torch.abs(param))
    return L1_reg
