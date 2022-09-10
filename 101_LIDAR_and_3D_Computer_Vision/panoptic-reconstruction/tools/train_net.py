import argparse
import os
from pathlib import Path

from lib.utils.debugger import Debugger

from lib import utils, logger, engine
from lib.config import config


def main() -> None:
    # arguments
    parser = argparse.ArgumentParser(description="Panoptic 3D Scene Reconstruction from a Single RGB Image")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # configuration
    config.OUTPUT_DIR = args.output_path

    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # basic paths
    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    output_config_path = output_path / "config.yaml"
    utils.save_config(config, output_config_path)
    utils.setup_logger(output_path, "log.txt")

    # output some basic information
    logger.info(args)
    logger.info("Collecting environment information...")
    logger.info(f"\n{utils.collect_env_info()}")
    logger.info(f"Loaded configuration file: {args.config_file}")
    config_file_content = open(args.config_file).read()
    logger.info(f"\n{config_file_content}")
    logger.info(f"Running with config:{config}")
    logger.info(f"Saving config at {output_config_path}")

    # make sure it's deterministic
    utils.re_seed()

    trainer = engine.Trainer()
    trainer.do_train()


if __name__ == '__main__':
    # Debugger()
    main()
