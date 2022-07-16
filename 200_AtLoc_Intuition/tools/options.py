import argparse
import os
from tools import utils
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        # base options
        self.parser.add_argument('--data_dir', type=str, default='./data')
        self.parser.add_argument('--batchsize', type=int, default=64)
        self.parser.add_argument('--cropsize', type=int, default=256)
        self.parser.add_argument('--print_freq', type=int, default=20)
        self.parser.add_argument('--gpus', type=str, default='-1')
        self.parser.add_argument('--nThreads', default=8, type=int, help='threads for loading data')
        self.parser.add_argument('--dataset', type=str, default='RobotCar')
        self.parser.add_argument('--scene', type=str, default='loop')
        self.parser.add_argument('--model', type=str, default='AtLoc')
        self.parser.add_argument('--seed', type=int, default=7)
        self.parser.add_argument('--lstm', type=bool, default=False)
        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')
        self.parser.add_argument('--skip', type=int, default=10)
        self.parser.add_argument('--variable_skip', type=bool, default=False)
        self.parser.add_argument('--real', type=bool, default=False)
        self.parser.add_argument('--steps', type=int, default=3)
        self.parser.add_argument('--val', type=bool, default=False)

        # train options
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--beta', type=float, default=-3.0)
        self.parser.add_argument('--gamma', type=float, default=None, help='only for AtLoc+ (-3.0)')
        self.parser.add_argument('--color_jitter', type=float, default=0.7, help='0.7 is only for RobotCar, 0.0 for 7Scenes')
        self.parser.add_argument('--train_dropout', type=float, default=0.5)
        self.parser.add_argument('--val_freq', type=int, default=5)
        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')
        self.parser.add_argument('--lr', type=float, default=5e-5)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)

        # test options
        self.parser.add_argument('--test_dropout', type=float, default=0.0)
        self.parser.add_argument('--weights', type=str, default='epoch_005.pth.tar')
        self.parser.add_argument('--save_freq', type=int, default=5)


    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpus.split(',')
        self.opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpus.append(id)

        # set gpu ids
        if len(self.opt.gpus) > 0:
            torch.cuda.set_device(self.opt.gpus[0])

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')

        # save to the disk
        self.opt.exp_name = '{:s}_{:s}_{:s}_{:s}'.format(self.opt.dataset, self.opt.scene, self.opt.model, str(self.opt.lstm))
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
        self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        utils.mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])
        return self.opt
