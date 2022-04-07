import argparse
import os
from models import find_model_class_by_name
from data import find_dataset_class_by_name
import torch


class BaseOptions:
    def initialize(self, parser: argparse.ArgumentParser):
        # ================================ global ================================#
        parser.add_argument(
            "--name", type=str, required=True, help="name of the experiment"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if specified, print more debugging information",
        )
        parser.add_argument(
            "--timestamp",
            action="store_true",
            help="suffix the experiment name with current timestamp",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed for numpy and torch, should be set right after option parsing",
        )

        # ================================ dataset ================================#
        parser.add_argument(
            "--data_root", type=str, required=True, help="path to the dataset storage"
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            required=True,
            help="name of dataset, determine which dataset class to use",
        )
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples allowed per dataset."
            "If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )
        parser.add_argument(
            "--n_threads", default=4, type=int, help="# threads for loading data"
        )

        # ================================ model ================================#
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="name of model, determine which network model to use",
        )

        # ================================ running ================================#
        parser.add_argument(
            "--batch_size", type=int, default=1, help="input batch size"
        )
        parser.add_argument(
            "--serial_batches",
            type=int,
            default=0,
            help="feed batches in order without shuffling",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        parser.add_argument(
            "--resume_dir", type=str, default="", help="dir of the previous checkpoint"
        )
        parser.add_argument(
            "--resume_epoch",
            type=str,
            default="latest",
            help="which epoch to resume from",
        )
        parser.add_argument("--debug", action="store_true", help="indicate a debug run")

        return parser

    def gather_options(self, modify):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        model_name = opt.model
        find_model_class_by_name(model_name).modify_commandline_options(
            parser, self.is_train
        )

        dataset_name = opt.dataset_name
        find_dataset_class_by_name(dataset_name).modify_commandline_options(
            parser, self.is_train
        )

        if modify:
            modify(parser)

        self.parser = parser

        return parser.parse_args()

    def print_and_save_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: {}]".format(str(default))
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        if opt.is_train:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        else:
            expr_dir = os.path.join(opt.resume_dir, opt.name)

        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, modify=None):
        opt = self.gather_options(modify)
        opt.is_train = self.is_train

        if opt.timestamp:
            import datetime

            now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
            opt.name = opt.name + "_" + now

        self.print_and_save_options(opt)

        opt.gpu_ids = [
            int(x) for x in opt.gpu_ids.split(",") if x.strip() and int(x) >= 0
        ]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
