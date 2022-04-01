# get commandline params before init
import argparse
import os


def parse_cmd_params():
    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument(
        "--config",
        "-c",
        dest="config_fn",
        required=True,
        help="(str) Config file name e.g. ./configs/test.yaml",
    )
    arg_parser.add_argument(
        "--gpu",
        "-g",
        type=str,
        dest="gpu",
        default=None,
        help="(str) GPU id to use, e.g --gpu=0,1,2,3 or --gpu=1; if not specify, use all gpus",
    )
    arg_parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        dest="batch_size",
        default=-1,
        help="(int) Batch size",
    )
    arg_parser.add_argument(
        "--resume",
        "-r",
        type=str,
        dest="resume",
        default=None,
        help="Provide the resume check point if resume, valid args are latest or 100,1000....",
    )
    arg_parser.add_argument(
        "--debug",
        "-d",
        dest="debug_logging_flag",
        default=False,
        action="store_true",
        help="(Bool) If set, print all debug information",
    )
    arg_parser.add_argument(
        "--anomaly",
        "-a",
        dest="enable_anomaly",
        default=False,
        action="store_true",
        help="(Bool) If set, use pytorch anomaly detection",
    )
    arg_parser.add_argument(
        "--force_continue",
        "-f",
        dest="no_interaction",
        default=False,
        action="store_true",
        help="(Bool) If set, will not use interactive confirm before start, used in cluster batch job",
    )
    args = arg_parser.parse_args()
    return args


def merge_cmd2cfg(cmd, cfg):
    """
    Add cmd line parameters to init
    """
    if int(cmd.batch_size) > 0:
        cfg['training']['batch_size'] = cmd.batch_size
    cfg['logging']['debug_mode'] = cmd.debug_logging_flag
    if cmd.gpu is not None:
        cfg['gpu'] = cmd.gpu
    if isinstance(cfg['gpu'], int):
        cfg['gpu'] = str(cfg['gpu'])
    if cmd.resume is not None:
        cfg['resume'] = cmd.resume
    cfg['logging']['loggers'] += ['checkpoint', 'metric']
    return cfg
