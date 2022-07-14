"""
Performs training, validation, testing for nvs_model.py and calculates loss and saves it to tensorboard.
Author: Lukas Hoellein
"""

import numpy as np

from models.synthesis.synt_loss_metric import SynthesisLoss, QualityMetrics, SynthesisLossRGBandSeg
from models.nvs_model import NovelViewSynthesisModel
import os
import re
from util.camera_transformations import invert_K, invert_RT

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from time import time
from tqdm.auto import tqdm

def to_cuda(data_tuple):
    out = ()
    if torch.cuda.is_available():
        for data in data_tuple:
            if isinstance(data, dict):
                for k, v in data.items():
                    data[k] = data[k].to("cuda:0")
                out += (data,)
            else:
                out += (data.to("cuda:0"),)
    return out

def default_batch_loader(batch):
    input_img = batch['image']
    K = batch['cam']['K']
    K_inv = batch['cam']['Kinv']
    input_RT = batch['cam']['RT1']
    input_RT_inv = batch['cam']['RT1inv']
    output_RT = batch['cam']['RT2']
    output_RT_inv = batch['cam']['RT2inv']
    gt_img = batch['output']['image'] if batch['output'] is not None else None
    input_seg = batch['seg']
    gt_seg = batch['output']['seg'] if batch['output'] is not None else None
    depth_img = batch['depth']
    dynamics = batch['dynamics']

    # this could also be None if such data is not present in the dataset
    gt_img_moved_for_evaluation_only = batch['output']['gt_moved_rgb_for_evaluation_only'] if batch['output'] is not None else None

    return input_img, K, K_inv, input_RT, input_RT_inv, output_RT, output_RT_inv, gt_img, input_seg, gt_seg, depth_img, dynamics, gt_img_moved_for_evaluation_only

# NOTE: Unused, might be used for debugging
def check_norm(img, verbose=False):
    """Try to determine the range of img and return the range in the form of: (left_end, right_end)"""
    max_val = torch.max(img)
    min_val = torch.min(img)

    if verbose:
        print("max_val:", max_val)
        print("min_val:", min_val)

    # Range: [0,1]
    if (0 <= min_val and min_val <= 1) and (0 <= max_val and max_val <= 1):
        return (0,1)

    # Range: [-1,1]
    elif (-1 <= min_val and min_val <= 1) and (-1 <= max_val and max_val <= 1):
        return (-1,1)

    # Range: [0,255]
    elif (0 <= min_val and min_val <= 255) and (0 <= max_val and max_val <= 255):
        return (0,255)

    # Unknown range
    else:
        print("WARNING: Input image doesn't seem to have values in ranges: [0,1], [-1,1], [0,255]")
        return None

# NOTE: Unused, might be used for debugging
def change_norm(img, in_range=None, out_range=[0,1]):
    """Based on the norm scheme of img and output the same image in the new norm scheme"""
    if not in_range:
        in_range = check_norm(img)

    img = (img - in_range[0]) / (in_range[1] - in_range[0]) * (out_range[1] - out_range[0]) + out_range[0]
    return img

class Checkpoint(object):
    def __init__(self,
                 model_args={},
                 model_name="",
                 solver_args={},
                 solver_name="",
                 checkpoint_path="",
                 checkpoint_freq=1):
        """

        """
        self.model_args = model_args
        self.model_name = model_name
        self.solver_args = solver_args
        self.solver_name = solver_name
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq if checkpoint_freq else 1

        self.max_val_acc = -np.inf
        self.last_saved = -1

    def print_config(self):
        print("Checkpoint configuration:\n",
              "Path: ", self.checkpoint_path, "\n",
              "Frequency: ", self.checkpoint_freq, "\n",
              "Model name: ", self.model_name, "\n",
              "Model arguments: ", self.model_args, "\n",
              "Solver arguments: ", self.solver_args, "\n",
              "Solver name: ", self.solver_name, "\n",
              "Maximum validation accuracy observed: ", self.max_val_acc, "\n",
              "Last checkpointed epoch: ", self.last_saved, sep="")

    def set_freq(self, checkpoint_freq):
        self.checkpoint_freq = checkpoint_freq

    def set_path(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def setup(self, start_epoch=0):
        """Folder setup and sanity checks for checkpoints"""
        try:
            os.mkdir(self.checkpoint_path)
            print("Folder {} is succesfully created".format(self.checkpoint_path))

        # Understand if it is possible to continue without overwriting the existing checkpoints
        except FileExistsError:
            files = os.listdir(self.checkpoint_path)                     # Read checkpoints
            # If there are already checkpoints saved previously
            if len(files) > 0:
                regex = r"_(\d+)\.pt"                                    # Regex for epoch
                func = lambda text: int(re.search(regex, text).group(1)) # Extract each epoch
                last_epoch = max(map(func, files))                       # Find last epoch of last run

                # Warn about overwriting possibility, ask for permission
                if start_epoch < last_epoch:
                    print(("WARNING: {} already exists.\n"+
                        "Given start_epoch ({}) may overwrite existing checkpoints (From previous run, last_epoch was {}).\n")
                        .format(self.checkpoint_path, start_epoch, last_epoch))

                    repsonse = input("Overwrite existing? [y/N]: ")
                    if repsonse.lower() != 'y':
                        print("Checkpointing disabled!")
                        self.checkpoint_freq = 0
            # If there are no checkpoints saved previously
            else:
                print("WARNING: {} already exists, but overwriting is not possible.".format(self.checkpoint_path))

        # Create checkpoints directory first
        except FileNotFoundError:
            print("Creating ../checkpoints/ folder...".format())
            os.mkdir("../checkpoints/")
            print("Creating {} folder...".format(self.checkpoint_path))
            os.mkdir(self.checkpoint_path)
            print("Folder {} is succesfully created".format(self.checkpoint_path))

    def save_checkpoint(self, model_state, optim_state, epoch, val_acc):
        """Called inside the training loop"""
        if self.checkpoint_freq > 0 and epoch % self.checkpoint_freq == 0:
            self.save(model_state, optim_state, epoch, val_acc)

    def save_final(self, model_state, optim_state, epoch, val_acc):
        """Called after the training loop"""
        # TODO: If train runs despite disabled checkpoint, with current implementation the final version of the model is saved. Is this a desired behaviour?
        if self.last_saved != epoch:
            self.save(model_state, optim_state, epoch, val_acc)

    def save(self, model_state, optim_state, epoch, val_acc):
        file_name = self.model_name + "_" + str(epoch) + ".pt" # TODO: Add id_suffix??
        full_path = os.path.join(self.checkpoint_path, file_name)
        # TODO: Except model_state, optim_state, epoch, val_acc, the rest of the entries are static during a run. We may save them once and reuse that file
        torch.save({'model_args': self.model_args,
                    'model_name': self.model_name,
                    'solver_args': self.solver_args, # TODO: solver_args includes extra_args, there are many duplicate entries
                    'solver_name': self.solver_name,
                    'model_state': model_state,
                    'optim_state': optim_state,
                    'start_epoch': epoch,
                    'max_val_acc': val_acc}, full_path)

        print("Checkpoint {} is created".format(file_name))

        self.last_saved = epoch
        self.max_val_acc = val_acc # TODO: change after implementing acc-based saving mechanism

    def load_checkpoint(self, path):
        saved_info = torch.load(path)
        # Reconstruct model from the skeleton
        model_args = saved_info["model_args"]
        model = NovelViewSynthesisModel(**model_args)

        # Reconstruct solver from the skeleton
        solver = None
        if saved_info["solver_name"] == "NVS_Solver":
            solver_args = saved_info["solver_args"]
            losses = solver_args.pop("loss_func")
            nvs_loss = SynthesisLoss(losses=losses)
            solver = NVS_Solver(**solver_args, loss_func=nvs_loss, optim_state=saved_info["optim_state"])
        else:
            # TODO: Support for GAN_Wrapper_Solver
            raise NotImplemented

        # Load model state, optim state is automatically loaded by solver
        model.load_state_dict(saved_info["model_state"])
        start_epoch = saved_info["start_epoch"]
        max_val_acc = saved_info["max_val_acc"]

        # Overwrite checkpoint state (Frequency should be reassigned)
        self.model_args = model_args
        self.model_name = saved_info["model_name"]
        self.solver_args = solver_args
        self.solver_name = saved_info["solver_name"]
        checkpoint_path = os.path.join(*path.split("/")[:-1])
        self.checkpoint_path = checkpoint_path
        self.max_val_acc = max_val_acc
        self.last_saved = start_epoch
        self.setup(start_epoch)

        return model, solver, start_epoch

class NVS_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam,
                 optim_state=None,
                 optim_args={},
                 loss_func=None,
                 extra_args={},
                 tensorboard_writer=None,
                 log_dir=None):
        """
        Parameters
        ----------
        optim: which optimizer to use, e.g. Adam
        optim_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        extra_args: extra_args that should be used when logging to tensorboard (e.g. model hyperparameters)
        tensorboard_writer: instance to use for writing to tensorboard. can be None, then a new one will be created.
        log_dir: where to log to tensorboard. Only used when no tensorboard_writer is given.
        """
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.optim_state = optim_state
        self.loss_func = loss_func if loss_func is not None else SynthesisLossRGBandSeg()
        self.acc_func = QualityMetrics()
        self.batch_loader = default_batch_loader

        self.writer = SummaryWriter(log_dir) if tensorboard_writer is None else tensorboard_writer

        for key in extra_args.keys():
            extra_args[key] = str(extra_args[key])
        self.hparam_dict = {'loss_function': type(self.loss_func).__name__,
                            'optimizer': self.optim.__name__,
                            'learning_rate': self.optim_args['lr'],
                            'weight_decay': self.optim_args['weight_decay'],
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

    def forward_pass(self, model, batch):

        batch = to_cuda(self.batch_loader(batch))

        # this gets not even passed to our model, only for evaluation purposes
        gt_img_moved_for_evaluation_only = batch[-1]
        batch = batch[:-1]

        output = model(*batch)
        dynamics = batch[-1]
        acc_dir = None

        if dynamics:
            loss_dir = self.loss_func(output['PredImg'], output['OutputImg'], output['PredSeg'], output['OutputSeg'], dynamics["input_mask"], dynamics["output_mask"])
            if self.acc_func is not None:
                acc_dir = self.acc_func(output['PredImg'], output['OutputImg'], output['PredSeg'], output['OutputSeg'], dynamics["output_mask"], dynamics["input_mask"])
        else:
            loss_dir = self.loss_func(output['PredImg'], output['OutputImg'], output['PredSeg'], output['OutputSeg'])
            if self.acc_func is not None:
                acc_dir = self.acc_func(output['PredImg'], output['OutputImg'], output['PredSeg'], output['OutputSeg'])

        if dynamics and gt_img_moved_for_evaluation_only is not None:
            # evaluate with gt_rgb_img_moved
            gt_img_moved_acc_dir = self.acc_func(output['PredImg'], gt_img_moved_for_evaluation_only, output['PredSeg'], output['OutputSeg'], dynamics["output_mask"], dynamics["input_mask"])

            # do not keep the segmentation values as they are identical to the ones above and add a prefix to the other ones
            gt_img_moved_acc_dir = {"gt_rgb_moved_"+k: v for k,v in gt_img_moved_acc_dir.items() if not "seg" in k}

            # merge this dict into acc_dir
            acc_dir.update(gt_img_moved_acc_dir)

        return loss_dir, output, acc_dir

    def log_iteration_loss_and_acc(self, loss_dir, acc_dir, prefix, idx): # acc_dir argument needed
        # WRITE LOSSES
        for loss in loss_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Loss/' + loss,
                                   loss_dir[loss].detach().cpu().numpy(),
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

    def visualize_output(self, output, take_slice=None, tag="image", step=0, depth=True):
        """
        Generic method for visualizing a single image or a whole batch
        Parameters
        ----------
        output: batch of data, containing input, target, prediction and depth image
        take_slice: two element tuple or list can be specified to take a slice of the batch (default: take whole batch)
        tag: used for grouping images on tensorboard. e.g. "train", "val", "test" etc.
        step: used for stamping epoch or iteration
        """
        # TODO: depth_batch is ignored for the moment, however, if needed, it can also be integrated later on
        input_batch, target_batch, target_batch_seg, pred_batch, pred_batch_seg, depth_batch, input_depth = output["InputImg"].detach().cpu(),\
                                                                          output["OutputImg"].detach().cpu(), \
                                                                          output["OutputSeg"].detach().cpu(), \
                                                                          output["PredImg"].detach().cpu(), \
                                                                          output["PredSeg"].detach().cpu(), \
                                                                          output["PredDepth"].detach().cpu(),\
                                                                          output['InputDepth'].detach().cpu()
        with torch.no_grad():
            # In case of a single image add one dimension to the beginning to create single image batch
            if len(pred_batch.shape) == 3:
                input_batch = input_batch.unsqueeze(0)
                target_batch = target_batch.unsqueeze(0)
                target_batch_seg = target_batch_seg.unsqueeze(0)
                pred_batch = pred_batch.unsqueeze(0)
                pred_batch_seg = pred_batch_seg.unsqueeze(0)
                depth_batch = depth_batch.unsqueeze(0)

            if len(pred_batch.shape) != 4:
                print("Only 3D or 4D tensors can be visualized")
                return

            # If slice specified, take a portion of the batch
            if take_slice and (type(take_slice) in (list, tuple)) and (len(take_slice) == 2):
                input_batch = input_batch[take_slice[0], take_slice[1]]
                target_batch = target_batch[take_slice[0], take_slice[1]]
                target_batch_seg = target_batch_seg[take_slice[0], take_slice[1]]
                pred_batch = pred_batch[take_slice[0], take_slice[1]]
                pred_batch_seg = pred_batch_seg[take_slice[0], take_slice[1]]
                depth_batch = depth_batch[take_slice[0], take_slice[1]]

            # Store vstack of images: [input_batch0, target_batch0, pred_batch0 ...].T on img_lst
            img_lst = torch.Tensor()

            # Run a loop to interleave images in input_batch, target_batch, pred_batch batches
            for i in range(pred_batch.shape[0]):
                # Each iteration pick input image and corresponding target & pred images
                # As we index image from batch, we need to extend the dimension of indexed images with .unsqueeze(0) for vstack
                # Order in img_list defines the layout.
                # Current layout: input - target - pred at each row
                img_lst = torch.cat((img_lst,
                    input_batch[i].unsqueeze(0),
                    target_batch[i].unsqueeze(0),
                    pred_batch[i].unsqueeze(0),
                    target_batch_seg[i].unsqueeze(0),
                    pred_batch_seg[i].unsqueeze(0)), dim=0)

            if depth:
                depth_lst = torch.Tensor()
                depth_batch = (depth_batch-0)/(10-0)
                input_depth = (input_depth-0)/(10-0)
                for i in range(depth_batch.shape[0]):
                    # Each iteration pick input image and corresponding target & pred images
                    # As we index image from batch, we need to extend the dimension of indexed images with .unsqueeze(0) for vstack
                    # Order in img_list defines the layout.
                    # Current layout: input - target - pred at each row
                    depth_lst = torch.cat((depth_lst,
                        input_depth[i].unsqueeze(0),
                        depth_batch[i].unsqueeze(0)), dim=0)
                depth_grid = make_grid(depth_lst, nrow=2)
                self.writer.add_image(tag+'/depth', depth_grid, global_step=step)

            img_grid = make_grid(img_lst, nrow=5) # Per row, pick five images from the stack
            # TODO: this idea can be extended, we can even parametrize this
            # TODO: if needed, determine range of values and use make_grid flags: normalize, range

            self.writer.add_image(tag, img_grid, global_step=step) # NOTE: add_image method expects image values in range [0,1]
            self.writer.flush()

    def test(self, model, test_loader, test_prefix='/', log_nth=0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if test_prefix == None:
            test_prefix = '/'
        elif not test_prefix.endswith('/'):
            test_prefix += '/'
        test_name = 'Test/' + test_prefix

        with torch.no_grad():
            test_losses = []
            test_accs = []
            for i, sample in enumerate(tqdm(test_loader)):
                loss_dir, output, test_acc_dir = self.forward_pass(model, sample)
                loss, test_acc = self.log_iteration_loss_and_acc(loss_dir,
                                                                 test_acc_dir,
                                                                 test_name,
                                                                 i)
                test_losses.append(loss)
                test_accs.append(test_acc)

                # Print loss every log_nth iteration
                if log_nth != 0 and i % log_nth == 0:
                    print("[Iteration {cur}/{max}] TEST loss: {loss}".format(cur=i + 1,
                                                                              max=len(test_loader),
                                                                              loss=loss))
                    self.visualize_output(output, tag="test", step=i)

            mean_loss = np.mean(test_losses)
            mean_acc = np.mean(test_accs)

            self.writer.add_scalar(test_name + 'Mean/Loss', mean_loss, 0)
            self.writer.add_scalar(test_name + 'Mean/Accuracy', mean_acc, 0)
            self.writer.flush()

            print("[TEST] mean acc/loss: {acc}/{loss}".format(acc=mean_acc, loss=mean_loss))

    def backward_pass(self, loss_dir, optim):
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss_dir['Total Loss'].backward()
        optim.step()
        optim.zero_grad()

        #print(prof)

    def train(self,
              model,
              train_loader,
              val_loader,
              checkpoint_handler=None,
              start_epoch=0,
              num_epochs=10,
              log_nth_iter=1,
              log_nth_epoch=1,
              tqdm_mode='total',
              verbose=False):
        """
        Train a given model with the provided data.
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
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Use registered optimizer state if available (becomes available when loaded from checkpoint.)
        if self.optim_state:
            optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()))
            optim.load_state_dict(self.optim_state)
        else:
            optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)

        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print('START TRAIN on device: {}'.format(device))

        max_epoch = start_epoch + num_epochs
        epochs = range(start_epoch, max_epoch)
        if tqdm_mode == 'total':
            epochs = tqdm(range(start_epoch, max_epoch))
        for epoch in epochs:  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []

            train_minibatches = train_loader
            if tqdm_mode == 'epoch':
                train_minibatches = tqdm(train_minibatches)

            # MEASURE ELAPSED TIME
            if verbose:
                # start first dataloading record
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            for i, sample in enumerate(train_minibatches):  # for every minibatch in training set
                # FORWARD PASS --> Loss + acc calculation

                # MEASURE ELAPSED TIME
                if verbose:
                    # end dataloading pass record
                    end.record()
                    torch.cuda.synchronize()
                    print("Dataloading took: {}".format(start.elapsed_time(end)))

                    # start forward/backward record
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                # FORWARD PASS
                train_loss_dir, train_output, train_acc_dir = self.forward_pass(model, sample)

                # BACKWARD PASS --> Gradient-Descent update
                self.backward_pass(train_loss_dir, optim)

                # LOGGING of loss and accuracy
                train_loss, train_acc = self.log_iteration_loss_and_acc(train_loss_dir,
                                                                        train_acc_dir,
                                                                        'Train/',
                                                                        epoch * iter_per_epoch + i)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # Print loss every log_nth iteration
                if log_nth_iter != 0 and (i+1) % log_nth_iter == 0:
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                              max=iter_per_epoch,
                                                                              loss=train_loss))
                    self.visualize_output(train_output, tag="train", step=epoch*iter_per_epoch + i)

                # MEASURE ELAPSED TIME
                if verbose:
                    # end forward/backward pass record
                    end.record()
                    torch.cuda.synchronize()
                    print("Forward/Backward Pass took: {}".format(start.elapsed_time(end)))

                    # start dataloading record
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

            # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)

            if log_nth_epoch != 0 and (epoch+1) % log_nth_epoch == 0:
                print("[EPOCH {cur}/{max}] TRAIN mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                     max=max_epoch,
                                                                                     acc=mean_train_acc,
                                                                                     loss=mean_train_loss))
                # If last iteration of the train epoch was not already logged
                num_iter = len(train_minibatches)
                if (log_nth_iter == 0) or (num_iter % log_nth_iter != 0):
                    self.visualize_output(train_output, tag="train", step=epoch*iter_per_epoch + i)

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
                        # FORWARD PASS --> Loss + acc calculation
                        val_loss_dir, val_output, val_acc_dir = self.forward_pass(model, sample)
                        val_loss, val_acc = self.log_iteration_loss_and_acc(val_loss_dir,
                                                                            val_acc_dir,
                                                                            'Val/',
                                                                            epoch * len(val_minibatches) + i)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)

                        # Print loss every log_nth iteration
                        if log_nth_iter != 0 and (i+1) % log_nth_iter == 0:
                            print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                                    max=len(val_loader),
                                                                                    loss=val_loss))
                            self.visualize_output(val_output, tag="val", step=epoch*len(val_minibatches) + i)

                    mean_val_loss = np.mean(val_losses)
                    mean_val_acc = np.mean(val_accs)

                    if log_nth_epoch != 0 and (epoch+1) % log_nth_epoch == 0:
                        print("[EPOCH {cur}/{max}] VAL mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                           max=max_epoch,
                                                                                           acc=mean_val_acc,
                                                                                           loss=mean_val_loss))
                        # If last iteration of the val epoch was not already logged
                        num_iter = len(val_minibatches)
                        if (log_nth_iter == 0) or (num_iter % log_nth_iter != 0):
                            self.visualize_output(val_output, tag="val", step=epoch*len(val_minibatches) + i)

            # LOG EPOCH LOSS / ACC FOR TRAIN AND VAL IN TENSORBOARD
            self.log_epoch_loss_and_acc(mean_train_loss, mean_val_loss, mean_train_acc, mean_val_acc, epoch)

            # Create a checkpoint
            if checkpoint_handler is not None:
                checkpoint_handler.save_checkpoint(model.state_dict(), optim.state_dict(), epoch+1, mean_val_acc)

        # Save the final model (if not already saved)
        if checkpoint_handler is not None:
            checkpoint_handler.save_final(model.state_dict(), optim.state_dict(), epoch+1, mean_val_acc)

        self.writer.add_hparams(self.hparam_dict, {
            'HParam/Accuracy/Val': self.val_acc_history[-1] if len(val_loader) > 0 else 0,
            'HParam/Accuracy/Train': self.train_acc_history[-1],
            'HParam/Loss/Val': self.val_loss_history[-1] if len(val_loader) > 0 else 0,
            'HParam/Loss/Train': self.train_loss_history[-1]
        })
        self.writer.flush()
        print('FINISH.')