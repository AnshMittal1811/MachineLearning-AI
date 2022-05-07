import hydra, logging
import torch, glob, os
import numpy as np
from trainers import *
from models import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="ConDor.yaml")
def run(cfg):

    seed_everything(cfg.utils.seed)
    train_logger = eval(cfg.logging.type)(project = cfg.logging.project)
    log.info(cfg)
    print(os.getcwd())
    checkpoint_callback = ModelCheckpoint(**cfg.callback.model_checkpoint.segmentation.args)
    model = getattr(eval(cfg.trainer_file.file), cfg.trainer_file.type)(configs = cfg)
    trainer = Trainer(**cfg.trainer, callbacks = [checkpoint_callback], logger = train_logger)
    trainer.fit(model)
    

if __name__ == '__main__':

    run()
