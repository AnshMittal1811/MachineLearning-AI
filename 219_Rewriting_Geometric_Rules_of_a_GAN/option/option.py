import argparse
import shlex
import os
import pickle

import util
import models
import datasets
import evaluation
import trainers
from util import Visualizer


class BaseOptions():
    def initialize(self, parser):
        # experiment specifics
        parser.add_argument("--disable_checkpoint", action="store_true", help='disable checkpointing, logging, generating webpages etc.')
        parser.add_argument('--num_gpus', type=int, default=1, help='#GPUs to use. 0 means CPU mode')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        parser.add_argument('--model', type=str, default='rewrite', help='which model to use')
        parser.add_argument('--trainer', type=str, default='warp', help='which model to use')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--resume_iter', type=str, default=None, help="# iterations (in thousands, or `latest`) to resume")
        parser.add_argument('--disable_eval', action='store_true')

        # input/output sizes
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--shuffle_dataset', type=str, default=None, choices=('true', 'false'))

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default=".")
        parser.add_argument('--dataset_mode', type=str, default='warp')
        parser.add_argument('--dataload_multiproc', action='store_true', help='use this flag to enable dataloader multiprocessing. This should be off for small datasets.')
        parser.add_argument('--max_train_samples', type=int, default=None, help='max number of training samples used')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        # networks
        parser.add_argument("--archG", default="stylegan3")
        parser.add_argument("--pretrained_G", required=True)

        # misc
        parser.add_argument('--cudnn_benchmark', default=True, type=bool, help='set torch.backends.cudnn.benchmark to True or not')
        parser.add_argument("--warp_cropped_car", action='store_true', help='handles the black region of StyleGAN cars')

        return parser

    def gather_options(self, command=None):
        parser = AugmentedArgumentParser()
        parser.custom_command = command

        # get basic options
        parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # update the parser according to the basic options
        parser = self.update_parser(parser, opt)

        # get the option with the updated parser
        opt, unknown = parser.parse_known_args()
        opt = parser.parse_args()

        # update the options
        self.parser = parser
        return opt

    def update_parser(self, parser, opt):
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # # modify network-related parser options
        # parser = networks.modify_commandline_options(parser, self.isTrain)

        # modify trainer-related parser options
        trainer_name = opt.trainer
        trainer_option_setter = trainers.get_option_setter(trainer_name)
        parser = trainer_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = datasets.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        # modify parser options related to iteration_counting
        parser = Visualizer.modify_commandline_options(parser, self.isTrain)

        # modify evaluation-related parser options
        evaluation_option_setter = evaluation.get_option_setter()
        parser = evaluation_option_setter(parser, self.isTrain)

        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def parse(self, save=False, command=None):
        opt = self.gather_options(command)
        opt.isTrain = self.isTrain   # train or test
        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        opt.dataroot = os.path.expanduser(opt.dataroot)

        assert opt.num_gpus <= opt.batch_size, "Batch size must not be smaller than num_gpus"
        return opt


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--continue_train', type=util.str2bool, default=False, help="resume training from last checkpoint")
        parser.add_argument('--pretrained_name', type=str, default=None,
                            help="Load weights from the checkpoint of another experiment")

        return parser


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to load samples and models')
        parser.add_argument("--result_dir", type=str, default="results")
        parser.add_argument("--model_path", type=str, default=None)
        return parser

    def parse(self, save=False, command=None):
        opt = super().parse(save, command)
        if opt.model_path is None:
            opt.model_path = os.path.join(opt.checkpoints_dir, opt.name, 'final_checkpoint.pth')
        elif opt.name is None:
            opt.name = os.path.splitext(os.path.basename(opt.model_path))[0]
        return opt


class AugmentedArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        """ Enables passing bash commands as arguments to the class.
        """
        print("parsing args...")
        if args is None and hasattr(self, 'custom_command') and self.custom_command is not None:
            print('using custom command')
            print(self.custom_command)
            args = shlex.split(self.custom_command)[2:]
        return super().parse_args(args, namespace)

    def parse_known_args(self, args=None, namespace=None):
        if args is None and hasattr(self, 'custom_command') and self.custom_command is not None:
            args = shlex.split(self.custom_command)[2:]
        return super().parse_known_args(args, namespace)

    def add_argument(self, *args, **kwargs):
        """ Support for providing a new argument type called "str2bool"

        Example:
        parser.add_argument("--my_option", type=util.str2bool, default=|bool|)

        1. "python train.py" sets my_option to be |bool|
        2. "python train.py --my_option" sets my_option to be True
        3. "python train.py --my_option False" sets my_option to be False
        4. "python train.py --my_option True" sets my_options to be True

        https://stackoverflow.com/a/43357954
        """

        if 'type' in kwargs and kwargs['type'] == util.str2bool:
            if 'nargs' not in kwargs:
                kwargs['nargs'] = "?"
            if 'const' not in kwargs:
                kwargs['const'] = True
        super().add_argument(*args, **kwargs)
