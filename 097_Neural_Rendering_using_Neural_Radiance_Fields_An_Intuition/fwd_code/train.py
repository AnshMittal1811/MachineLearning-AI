import os
import signal
import time

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter as tensorboardWriter
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.train_options import (
    ArgumentParser,
    get_log_path,
    get_model_path,
    get_timestamp,
    log_opt,
)
torch.backends.cudnn.benchmark = True

def train(epoch, data_loader, model, log_path, plotter, opts, is_save=True, debug_path=None):
    print("At train", flush=True)

    losses = {}
    iter_data_loader = iter(data_loader)

    for iteration in range(0, min(501, len(data_loader))):
        t_losses, output_image = model(
            iter_data_loader, isval=False, num_steps=opts.num_accumulations
        )

        for l in t_losses.keys():
            if l in losses.keys():
                losses[l] = t_losses[l].cpu().mean().detach().item() + losses[l]
            else:
                losses[l] = t_losses[l].cpu().mean().detach().item()
        if (iteration % 250 == 0 or iteration == 0) and is_save:
            for add_im in output_image.keys():
                if iteration == 0:
                    if not torch.is_tensor(output_image[add_im]):
                        num_views = len(output_image[add_im])
                        for index_view in range(num_views):
                            torchvision.utils.save_image(
                            output_image[add_im][index_view][0:8, :, :, :].cpu().data,
                            os.path.join(debug_path, "%d_%d_%s_view_%d.png" % (epoch, iteration, add_im, index_view)),
                            normalize=("Depth" in add_im),
                            )
                    else:
                        torchvision.utils.save_image(
                            output_image[add_im][0:8, :, :, :].cpu().data,
                            os.path.join(debug_path, "%d_%d_%s.png" % (epoch, iteration, add_im)),
                            normalize=("Depth" in add_im),
                        )
                if not torch.is_tensor(output_image[add_im]):
                    num_views = len(output_image[add_im])
                    for index_view in range(num_views):
                        plotter.add_image(
                            "Image_train/%d_%d_%s_view%d" % (epoch, iteration, add_im, index_view),
                            torchvision.utils.make_grid(
                                output_image[add_im][index_view][0:8, :, :, :].cpu().data,
                                normalize=("Depth" in add_im),
                            ),
                            epoch,
                        )
                else:
                    plotter.add_image(
                        "Image_train/%d_%d_%s" % (epoch, iteration, add_im),
                        torchvision.utils.make_grid(
                            output_image[add_im][0:8, :, :, :].cpu().data,
                            normalize=("Depth" in add_im),
                        ),
                        epoch,
                    )

        if iteration % 1 == 0:
            str_to_print = "Train: Epoch {}: {}/{} with ".format(
                epoch, iteration, len(data_loader)
            )
            for l in losses.keys():
                str_to_print += " %s : %0.4f | " % (
                    l,
                    losses[l] / float(iteration+1),
                )
            print(str_to_print, flush=True)


        for l in t_losses.keys():
            plotter.add_scalars(
                "%s_iter" % l,
                {"train": t_losses[l].cpu().mean().detach().item()},
                epoch * 500 + iteration,
            )
    if opts.lr_annealing:
        if epoch > opts.anneal_start:
            scale_factor = 1.0 * opts.anneal_factor ** (epoch // opts.anneal_t)
            model.lr_annealing(scale_factor)
    return {l: losses[l] / float(iteration) for l in losses.keys()}


def val(epoch, data_loader, model, log_path, plotter):

    losses = {}

    iter_data_loader = iter(data_loader)
    for iteration in range(0, min(501, len(data_loader))):
        t_losses, output_image = model(
            iter_data_loader, isval=True, num_steps=1
        )
        for l in t_losses.keys():
            if l in losses.keys():
                losses[l] = t_losses[l].cpu().mean().item() + losses[l]
            else:
                losses[l] = t_losses[l].cpu().mean().item()
        if iteration % 100 == 0 or iteration == 0:
            for add_im in output_image.keys():
                
                plotter.add_image(
                    "Image_val/%d_%s" % (iteration, add_im),
                    torchvision.utils.make_grid(
                        output_image[add_im][0:8, :, :, :].cpu().data,
                        normalize=("Depth" in add_im),
                    ),
                    epoch,
                )

        if iteration % 1 == 0:
            str_to_print = "Val: Epoch {}: {}/{} with ".format(
                epoch, iteration, len(data_loader)
            )
            for l in losses.keys():
                str_to_print += " %s : %0.4f | " % (
                    l,
                    losses[l] / float(iteration + 1),
                )
            print(str_to_print, flush=True)

        for l in t_losses.keys():
            plotter.add_scalars(
                "%s_iter" % l,
                {"val": t_losses[l].cpu().mean().detach().item()},
                epoch * 50 + iteration,
            )

    return {l: losses[l] / float(iteration + 1) for l in losses.keys()}


def checkpoint(model, save_path, CHECKPOINT_tempfile):
    if model.use_discriminator:
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizerG": model.optimizer_G.state_dict(),
            "epoch": model.epoch,
            "optimizerD": model.optimizer_D.state_dict(),
            "opts": opts,
        }

    else:
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizerG": model.optimizer_G.state_dict(),
            "epoch": model.epoch,
            "opts": opts,
        }

    torch.save(checkpoint_state, CHECKPOINT_tempfile)
    if os.path.isfile(CHECKPOINT_tempfile):
        os.rename(CHECKPOINT_tempfile, save_path)


def run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile, debug_path, opts):
    print("Starting run...", flush=True)

    opts.best_epoch = 0
    opts.best_loss = -1000
    if os.path.exists(opts.model_epoch_path) and opts.resume:
        past_state = torch.load(opts.model_epoch_path)
        print("Continuing epoch ... %d" % (past_state['opts'].continue_epoch + 1), flush=True)
        model.load_state_dict(torch.load(opts.model_epoch_path)["state_dict"])
        if opts.discriminator_losses:
            model.optimizer_D.load_state_dict(
                torch.load(opts.model_epoch_path)["optimizerD"]
            )
        model.optimizer_G.load_state_dict(
            torch.load(opts.model_epoch_path)["optimizerG"]
        )

        opts.continue_epoch = past_state["opts"].continue_epoch + 1
        opts.current_episode_train = past_state["opts"].current_episode_train
        opts.current_episode_val = past_state["opts"].current_episode_val
        opts.best_epoch = past_state["opts"].best_epoch
        opts.best_loss = past_state["opts"].best_loss
    elif opts.resume:
        print("WARNING: Model path does not exist?? ")
        print(opts.model_epoch_path)

    print("Loading train dataset ....", flush=True)
    train_set = Dataset("train", opts)

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    print("Loaded train dataset ...", flush=True)

    for epoch in range(opts.continue_epoch, opts.max_epoch):
        print("Starting epoch %d" % epoch, flush=True)
        opts.continue_epoch = epoch
        model.epoch = epoch
        model.train()

        if epoch % opts.val_period == 0:
            is_save = True
        else:
            is_save = False 

        train_loss = train(
            epoch, train_data_loader, model, log_path, plotter, opts, is_save, debug_path
        )
        if epoch % opts.val_period == 0:
            with torch.no_grad():

                # model.eval()
                train_set.toval(
                    epoch=0
                )  # Hack because don't want to keep reloading the environments
                loss = val(epoch, train_data_loader, model, log_path, plotter)
                train_set.totrain(epoch=epoch + 1 + opts.seed)

            for l in train_loss.keys():
                if l in loss.keys():
                    plotter.add_scalars(
                        "%s_epoch" % l,
                        {"train": train_loss[l], "val": loss[l]},
                        epoch,
                    )
                else:
                    plotter.add_scalars(
                        "%s_epoch" % l, {"train": train_loss[l]}, epoch
                    )

            if loss["psnr"] > opts.best_loss:
                checkpoint(
                    model, opts.model_epoch_path + "best", CHECKPOINT_tempfile
                )
                opts.best_epoch = epoch
                opts.best_loss = loss["psnr"]

            checkpoint(model, opts.model_epoch_path, CHECKPOINT_tempfile)

        if epoch % opts.val_period == 0:
            checkpoint(
                model,
                opts.model_epoch_path + "ep%d" % epoch,
                CHECKPOINT_tempfile,
            )

    if epoch == 100 - 1:
        open(HALT_filename, "a").close()


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    opts, _ = ArgumentParser().parse()
    opts.isTrain = True
    timestamp = get_timestamp()
    print("Timestamp ", timestamp, flush=True)
    print(opts.model_epoch_path)
    opts.model_epoch_path = get_model_path(timestamp, opts)
    print("Model ", opts.model_epoch_path, flush=True)

    Dataset = get_dataset(opts)
    model = get_model(opts)

    log_path = get_log_path(timestamp, opts)
    print(log_path)
    plotter = tensorboardWriter(logdir=log_path % "tensorboard")

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else:
        model = nn.DataParallel(model, torch_devices).to(device)

    CHECKPOINT_tempfile = opts.model_epoch_path + ".tmp"
    global CHECKPOINT_filename
    CHECKPOINT_filename = CHECKPOINT_tempfile
    HALT_filename = "HALT"
    opts.debug_path = log_path
    debug_path = os.path.join(opts.debug_path, "Image_train")
    
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    log_opt(debug_path, opts)
    if os.path.isfile(CHECKPOINT_tempfile):
        os.remove(CHECKPOINT_tempfile)

    if opts.load_old_model:
        model = BaseModel(model, opts)

        # Allow for different image sizes
        state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in torch.load(opts.old_model)["state_dict"].items()
            if not ("xyzs" in k) and not ("ones" in k)
        }
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
        run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile, debug_path, opts)
    else:
        model = BaseModel(model, opts)
        run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile, debug_path, opts)
