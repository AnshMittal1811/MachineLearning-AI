"""
Performs training, validation, testing for nvs_model.py with a GAN and calculates loss and saves it to tensorboard.

Author: Lukas Hoellein
"""

import numpy as np

from util.nvs_solver import NVS_Solver
from models.gan.gan_loss import DiscriminatorLoss

import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm.auto import tqdm

class GAN_Wrapper_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim_d=torch.optim.Adam,
                 optim_d_args={},
                 optim_g=torch.optim.Adam,
                 optim_g_args={},
                 g_loss_func=None, # if left None, the NVS_Solver will instantiate a standard SynthesisLoss class
                 extra_args={},
                 log_dir=None,
                 num_D=3,
                 size_D=64,
                 loss_D='ls',
                 no_gan_feature_loss=False,
                 lr_step=10,
                 lr_gamma=0.1,
                 log_depth=False,
                 init_discriminator_weights=True):
        """

        Parameters
        ----------
        optim_d: which optimizer to use for discriminator, e.g. Adam
        optim_d_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        optim_g: which optimizer to use for generator, e.g. Adam
        optim_g_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        extra_args: extra_args that should be used when logging to tensorboard (e.g. model hyperparameters)
        log_dir: where to log to tensorboard
        num_D: number of discriminators to use, every discriminator will work with lower resolution.
        e.g. One discriminator at full resolution and one with input downsampled to half the resolution
        size_D: number of channels/filters in each Conv2d in discriminator
        loss_D: the loss that the discriminator uses, options are ls (MSE-Loss), hinge, wgan (Wasserstein-GAN)
        or original (Cross-Entropy)
        init_discriminator_weights: if weights of the discriminator should be initialized
        """
        optim_d_args_merged = self.default_adam_args.copy()
        optim_d_args_merged.update(optim_d_args)
        self.optim_d_args = optim_d_args_merged
        self.optim_d = optim_d
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.log_depth = log_depth


        self.writer = SummaryWriter(log_dir)

        self.netD = DiscriminatorLoss(optim_d_args['lr'],
                                      gan_mode=loss_D,
                                      no_ganFeat_loss=no_gan_feature_loss,
                                      num_D=num_D,
                                      ndf=size_D,
                                      init=init_discriminator_weights)

        self.optimizer_D = self.optim_d(
            filter(lambda p: p.requires_grad, self.netD.parameters()),
            **self.optim_d_args
        )

        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D,
                                                           step_size=self.lr_step,
                                                           gamma=self.lr_gamma)

        self.nvs_solver = NVS_Solver(optim=optim_g,
                                     optim_args=optim_g_args,
                                     loss_func=g_loss_func,
                                     extra_args={},
                                     tensorboard_writer=self.writer)

        nvs_solver_args = {'generator_'+k: str(v) for k, v in self.nvs_solver.hparam_dict.items()}
        for key in extra_args.keys():
            extra_args[key] = str(extra_args[key])
        self.hparam_dict = {'discriminator': type(self.netD).__name__,
                            'discriminator_optim': self.optim_d.__name__,
                            'discriminator_learning_rate': self.optim_d_args['lr'],
                            **nvs_solver_args,
                            **extra_args}

        print("Hyperparameters of this solver: {}".format(self.hparam_dict))

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []


    def log_iteration_loss_and_acc(self, loss_dir, acc_dir, prefix, idx): # acc_dir argument needed
        # WRITE LOSSES
        for loss in loss_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Loss/' + loss,
                                   loss_dir[loss].detach().cpu().numpy(),
                                   idx)
        if 'D_Fake' and 'GAN' in loss_dir.keys():
            self.writer.add_scalars(prefix + 'Batch/CombinedLoss/',
                                    {'GAN': loss_dir['GAN'].detach().cpu().numpy(),
                                      'D_Fake': loss_dir['D_Fake'].detach().cpu().numpy()},
                                    idx)
        self.writer.flush()

        # WRITE ACCS
        for acc in acc_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Accuracy/' + acc,
                                   acc_dir[acc].detach().data.cpu().numpy(),
                                   idx)
        return loss_dir['Total Loss'].detach().cpu().numpy(), acc_dir["rgb_ssim"].detach().cpu().numpy() # could also use acc_dir["rgb_psnr"]


    def log_epoch_loss_and_acc(self, train_loss, val_loss, train_acc, val_acc, idx):
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.writer.add_scalar('Epoch/Loss/Train', train_loss, idx)
        self.writer.add_scalar('Epoch/Accuracy/Train', train_acc, idx)

        if val_loss is not None:
            self.val_loss_history.append(val_loss)
            self.writer.add_scalar('Epoch/Loss/Val', val_loss, idx)
            self.writer.add_scalars('Epoch/Loss',
                                    {'train': train_loss,
                                     'val': val_loss},
                                    idx)

        if val_acc is not None:
            self.val_acc_history.append(val_acc)
            self.writer.add_scalar('Epoch/Accuracy/Val', val_acc, idx)
            self.writer.add_scalars('Epoch/Accuracy',
                                    {'train': train_acc,
                                     'val': val_acc},
                                    idx)

        self.writer.flush()


    def train(self,
              model,
              train_loader,
              val_loader,
              num_epochs=10,
              log_nth_iter=1,
              log_nth_epoch=1,
              tqdm_mode='total',
              steps=1):
        """
        Train a given nvs_model with the provided data.

        Inputs:
        :param model: nvs_model object initialized from nvs_model.py
        :param train_loader: train data in torch.utils.data.DataLoader
        :param val_loader: val data in torch.utils.data.DataLoader
        :param num_epochs: total number of training epochs
        :param log_nth_iter: log training accuracy and loss every nth iteration. Default 1: meaning "Log everytime", 0 means "never log"
        :param log_nth_epoch: log training accuracy and loss every nth epoch. Default 1: meaning "Log everytime", 0 means "never log"
        :param tqdm_mode:
                'total': tqdm log how long all epochs will take,
                'epoch': tqdm for each epoch how long it will take,
                anything else, e.g. None: do not use tqdm
        :param steps: how many generator/discriminator steps to take before changing to discriminator/generator
        """

        optimizer_G = self.nvs_solver.optim(
            filter(lambda p: p.requires_grad, model.parameters()),
            **self.nvs_solver.optim_args
        )

        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                                      step_size=self.lr_step,
                                                      gamma=self.lr_gamma)

        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.netD.to(device)
        weight = 1.0 / float(steps)

        print('START TRAIN on device: {}'.format(device))

        #start = time()
        epochs = range(num_epochs)
        if tqdm_mode == 'total':
            epochs = tqdm(range(num_epochs))

        for epoch in epochs:  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []

            train_minibatches = train_loader
            if tqdm_mode == 'epoch':
                train_minibatches = tqdm(train_minibatches)
            for i, sample in enumerate(train_minibatches):  # for every minibatch in training set
                # RUN GENERATOR steps times
                all_output_images = []
                for j in range(0, steps):
                    nvs_losses, output, nvs_accs = self.nvs_solver.forward_pass(model, sample)
                    g_losses = self.netD.run_generator_one_step(output["PredImg"], output["OutputImg"])
                    (
                            g_losses["Total Loss"] / weight
                            + nvs_losses["Total Loss"] / weight
                    ).mean().backward()
                    all_output_images += [output]
                optimizer_G.step() # TODO: Why step after loop and not during every loop run?
                optimizer_G.zero_grad()

                # RUN DISCRIMINATOR steps times
                for step in range(0, steps):
                    d_losses = self.netD.run_discriminator_one_step(
                        all_output_images[step]["PredImg"],
                        all_output_images[step]["OutputImg"],
                    )
                    (d_losses["Total Loss"] / weight).mean().backward()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

                # UPDATE LOSS: nvs_losses contains G_LOSS and D_LOSS
                # TODO why not combine Total Loss to have nvs_loss + g_loss for logging (gets done for backward pass anyways?
                g_losses.pop("Total Loss")
                d_losses.pop("Total Loss")
                nvs_losses.update(g_losses)
                nvs_losses.update(d_losses)

                # LOGGING of loss
                train_loss, train_acc = self.log_iteration_loss_and_acc(nvs_losses, nvs_accs, 'Train/', epoch * iter_per_epoch + i)
                train_losses.append(train_loss) # TODO is this correct? see above: why not combine everything into Total Loss?
                train_accs.append(train_acc)

                # Print loss every log_nth iteration
                if log_nth_iter != 0 and (i+1) % log_nth_iter == 0:
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                              max=iter_per_epoch,
                                                                              loss=train_loss))
                    self.nvs_solver.visualize_output(all_output_images[-1], tag="train", step=epoch*iter_per_epoch + i, depth=self.log_depth)

            self.scheduler_D.step()
            scheduler_G.step()

            # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)

            if log_nth_epoch != 0 and (epoch+1) % log_nth_epoch == 0:
                print("[EPOCH {cur}/{max}] TRAIN mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                     max=num_epochs,
                                                                                     acc=mean_train_acc,
                                                                                     loss=mean_train_loss))
                self.nvs_solver.visualize_output(all_output_images[-1], tag="train", step=epoch*iter_per_epoch + i, depth=self.log_depth)

            # ONE EPOCH PASSED --> calculate + log validation accuracy/loss for this epoch
            mean_val_loss = None
            mean_val_acc = None
            if len(val_loader) > 0:
                model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
                with torch.no_grad():
                    val_losses = []
                    val_accs = []
                    val_minibatches = val_loader
                    if tqdm_mode == 'epoch':
                        val_minibatches = tqdm(val_minibatches)
                    for i, sample in enumerate(val_minibatches):

                        nvs_losses, output, nvs_accs = self.nvs_solver.forward_pass(model, sample)
                        val_loss, val_acc = self.log_iteration_loss_and_acc(nvs_losses, nvs_accs, 'Val/', epoch * len(val_minibatches) + i)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)

                        # Print loss every log_nth iteration
                        if log_nth_iter != 0 and (i+1) % log_nth_iter == 0:
                            print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                                    max=len(val_loader),
                                                                                    loss=val_loss))
                            self.nvs_solver.visualize_output(output, tag="val", step=epoch*len(val_minibatches) + i, depth=self.log_depth)

                    mean_val_loss = np.mean(val_losses)
                    mean_val_acc = np.mean(val_accs)

                    if log_nth_epoch != 0 and (epoch+1) % log_nth_epoch == 0:
                        print("[EPOCH {cur}/{max}] VAL mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                           max=num_epochs,
                                                                                           acc=mean_val_acc,
                                                                                           loss=mean_val_loss))
                        self.nvs_solver.visualize_output(output, tag="val", step=epoch*len(val_minibatches) + i, depth=self.log_depth)

            # LOG EPOCH LOSS / ACC FOR TRAIN AND VAL IN TENSORBOARD
            self.log_epoch_loss_and_acc(mean_train_loss, mean_val_loss, mean_train_acc, mean_val_acc, epoch)

        self.writer.add_hparams(self.hparam_dict, {
            'HParam/Accuracy/Val': self.val_acc_history[-1] if len(val_loader) > 0 else 0,
            'HParam/Accuracy/Train': self.train_acc_history[-1],
            'HParam/Loss/Val': self.val_loss_history[-1] if len(val_loader) > 0 else 0,
            'HParam/Loss/Train': self.train_loss_history[-1]
        })
        self.writer.flush()
        print('FINISH.')
