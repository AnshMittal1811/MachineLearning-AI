from src import training
from src.dataloader import IFNetDataModule
import argparse
import torch
import logging
from datetime import datetime
import os
import config.config_loader as cfg_loader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only


@rank_zero_only
def setup_logging(log_path):
    plain_formatter = logging.Formatter(
        "[%(asctime)s] Vis4D %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    logger = logging.getLogger("pytorch_lightning")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)


def main_train():

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(
        description='Train Model'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--version', type=str, default=None,
                        help='Version of the experiment.')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus used.')

    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from ckpt.')
    parser.add_argument('--weights', type=str,
                        default=None, help='Path of ckpt.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode.')
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    cfg = cfg_loader.load(args.config)

    time_stamp = datetime.now().strftime('%m-%d_%H-%M-%S')
    if args.version is None:
        args.version = time_stamp
    exp_name = cfg['model']

    if args.debug:
        exp_name += "_debug"

    output_dir = os.path.join(
        "experiments", exp_name, args.version
    )

    # setup logging
    log_path = os.path.join(output_dir, f"log_{time_stamp}.txt")
    setup_logging(log_path)

    rank_zero_info(cfg)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        verbose=True,
        filename="best",
        mode="min",
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=True,
    )

    resume_path = None
    if args.resume:  # pragma: no cover
        if args.weights is not None:
            resume_path = args.weights
        elif os.path.exists(os.path.join(output_dir, "checkpoints/last.ckpt")):
            resume_path = os.path.join(output_dir, "checkpoints/last.ckpt")

    datamodule = IFNetDataModule(cfg)
    module = training.get_trainers()[cfg['trainer']](cfg)

    # start logging
    if not args.debug:
        logger = WandbLogger(
            save_dir="experiments", name=f"{exp_name}-{args.version}", project=cfg['exp_name'])
    else:
        logger = None
    trainer = pl.Trainer(
        gpus=args.gpus if torch.cuda.is_available() else 0,
        max_epochs=100,
        logger=logger,
        resume_from_checkpoint=resume_path,
        callbacks=[checkpoint],
        plugins=DDPPlugin(
            find_unused_parameters=False) if args.gpus > 1 else None,
        accelerator="DDP" if args.gpus > 1 else None,
        num_sanity_val_steps=0,
        # gradient_clip_val=10
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main_train()
