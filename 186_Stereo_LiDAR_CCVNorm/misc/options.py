from easydict import EasyDict
import torch
import torch.nn as nn

from model import loss


def get_config():
    cfg = EasyDict()
    cfg.workspace = './exp/test' # Name of the workspace for an experiment
    cfg.cuda = True # Use GPU or not
    cfg.multi_gpu = False # Use multi-GPU or not; can only be set to True if batch size > 1
    cfg.to_disparity = True # Whether convert depth to disparity; mostly set to True
    cfg.dataset_name = 'kitti2017' # Dataset, for more settings please refer to `get_dataset` function in the below; kitti2017 / kitti2015
    cfg.batch_size = 1 # Batch size
    cfg.workers = 8 # Number of workers for dataloader
    cfg.model_name = 'gcnet_lidar' # Model, for more settings please refer to `get_model` function in the below; gcnet / gcnet_lidar
    cfg.criterion_name = 'l1' # Loss function, for more settings please refer to `get_criterion` function in the below
    cfg.optimizer_name = 'rmsprop' # Optimizer, for more settings please refer to `get_optimizer` function in the below
    cfg.lr_scheduler_name = None # Set to None for not using learning rate decay, see `get_lr_scheduler`
    cfg.pretrained = None # Set to None for not loading pretrained model
    cfg.weight_only = False # if pretrained model loaded, load model weight only or optimizer as well
    cfg.max_epoch = 10 # Number of epochs for training
    cfg.print_step = 1 # Print training results, unit: iteration
    cfg.tflog_step = 100 # Tensorboard log, unit: iteration
    cfg.val_step = 1000 # Online validation, unit: iteration
    cfg.save_step = 1000 # Saving checkpoints, unit: iteration
    cfg.train_metric_field = ['rmse', 'mae', 'mre', 'err_3px', 'err_2px', 'err_1px']
    cfg.val_metric_field = ['rmse', 'mae', 'mre', 'irmse', 'imae'] # for depth metric
    cfg.dump_all_param = False # Dump variables and gradients for all model parameters.
                               # NOTE: be careful, setting this to True will require a lot of HDD memory

    return cfg


def get_dataset(name):
    cfg = get_config()
    if name == 'kitti2017':
        from dataset.dataset_kitti2017 import DatasetKITTI2017
        rgb_dir = './data/kitti2017/rgb'
        depth_dir = './data/kitti2017/depth'
        exlude_data2015 = True # Set to True if you want to filter out sequences overlapped with KITTI2015
        train_output_size = (256, 512)
        val_output_size = (256, 960) # NOTE: set to (256, 1216) if there is enough gpu memory
        val_subset_size = 1000 # number of examples random sampled from the validation set
        train_dataset = DatasetKITTI2017(rgb_dir, depth_dir, 'train',
                                         train_output_size, to_disparity=cfg.to_disparity, exlude_data2015=exlude_data2015)
        val_dataset = DatasetKITTI2017(rgb_dir, depth_dir, 'val',
                                       val_output_size, to_disparity=cfg.to_disparity,
                                       use_subset=val_subset_size)
    else:
        raise NameError('Invalid dataset name {}'.format(name))

    return train_dataset, val_dataset


def get_model(name): # NOTE
    if name == 'gcnet':
        from model.gcnet import GCNet
        max_disparity = 192
        model = GCNet(max_disparity)
    elif name == 'gcnet_lidar':
        from model.gcnet_lidar import GCNetLiDAR
        max_disparity = 192
        norm_mode = ['naive_categorical', # Applying categorical CBN on 3D-CNN in stereo matching network
                     'naive_continuous', # Applying continuous CBN on 3D-CNN in stereo matching network
                     'categorical', # Applying categorical CCVNorm on 3D-CNN in stereo matching network
                     'continuous', # Applying continuous CCVNorm on 3D-CNN in stereo matching network
                     'categorical_hier', # Applying categorical HierCCVNorm on 3D-CNN in stereo matching network
                     ][4]
        model = GCNetLiDAR(max_disparity, norm_mode)
    else:
        raise NameError('Invalid model name {}'.format(name))

    return model


def get_criterion(name):
    if name == 'l1':
        criterion = loss.L1Loss()
    elif name == 'inv_disp_l1':
        criterion = loss.InvDispL1Loss()
    elif name == 'l2':
        criterion = loss.L2Loss()
    else:
        raise NameError('Invalid criterion name {}'.format(name))

    return criterion


def get_optimizer(name, params):
    if name == 'sgd':
        optim = torch.optim.SGD(params,
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1E-4)
    elif name == 'adam':
        optim = torch.optim.Adam(params,
                                 lr=1E-3,
                                 betas=(0.9, 0.999))
    elif name == 'rmsprop':
        optim = torch.optim.RMSprop(params,
                                    lr=1E-3,
                                    alpha=0.9)
    else:
        raise NameError('Invalid optimizer name {}'.format(name))

    return optim


def get_lr_scheduler(name, optim):
    if name is None:
        lr_scheduler = None
    elif name == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                       step_size=5,
                                                       gamma=0.2)
    else:
        raise NameError('Invalid learning rate schedular name {}'.format(name))

    return lr_scheduler
