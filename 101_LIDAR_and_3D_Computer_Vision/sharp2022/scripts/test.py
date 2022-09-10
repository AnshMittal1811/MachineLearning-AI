import argparse
import config.config_loader as cfg_loader
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from datetime import datetime
from src import training
from src.dataloader import IFNetDataModule
from src.generation import GeometryGenerationCallback, TextureGenerationCallback, PoseGenerationCallback


def main_generate():
    parser = argparse.ArgumentParser(description='Run generation')

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--weights', type=str,
                        default=None, help='Path of ckpt.')
    parser.add_argument('--version', type=str, default=None,
                        help='Version of the experiment.')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus used.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    if args.version is None:
        args.version = datetime.now().strftime('%m-%d_%H-%M-%S')
    exp_name = cfg['model']
    output_dir = os.path.join(
        "experiments", exp_name, args.version
    )

    datamodule = IFNetDataModule(cfg)
    module = training.get_trainers()[cfg['trainer']].load_from_checkpoint(
        args.weights, cfg=cfg)

    if cfg['action'] == "geometry":
        gen_callback = GeometryGenerationCallback(cfg, output_dir)
    elif cfg['action'] == "texture":
        gen_callback = TextureGenerationCallback(cfg, output_dir)
    elif cfg['action'] == "pose":
        gen_callback = PoseGenerationCallback(cfg, output_dir)

    trainer = pl.Trainer(
        gpus=args.gpus if torch.cuda.is_available() else 0,
        logger=None,
        callbacks=[gen_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        accelerator="DDP"
    )

    trainer.test(module, datamodule)


if __name__ == "__main__":
    main_generate()
