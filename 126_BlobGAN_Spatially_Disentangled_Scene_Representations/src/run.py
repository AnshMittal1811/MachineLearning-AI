import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import data
import models
import utils
from utils import is_rank_zero, scale_logging_rates


@hydra.main(config_path="configs", config_name="fit")
def run(config: DictConfig):
    torch.backends.cudnn.deterministic = config.trainer.deterministic
    torch.backends.cudnn.benchmark = config.trainer.benchmark
    torch.use_deterministic_algorithms(config.trainer.deterministic)
    if config.trainer.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if is_rank_zero():
        print(OmegaConf.to_yaml(config, resolve=True))

    seed_everything(config.seed, workers=True)

    scale_logging_rates(config, 1 / config.trainer.get('accumulate_grad_batches', 1))

    if config.get('detect_anomalies', False):
        if is_rank_zero():
            print('Anomaly detection mode ACTIVATED')
        torch.autograd.set_detect_anomaly(True)

    config.resume.id = utils.resolve_resume_id(**config.resume)

    if config.logger:
        logger = utils.Logger(**config[config.logger])
        logger.log_config(config)
        logger.log_code()
    else:
        logger = False

    datamodule = data.get_datamodule(**config.dataset)

    model, model_cfg = models.get_model(**config.model, return_cfg=True)

    if config.resume.id is not None:
        checkpoint = utils.get_checkpoint_path(**config.resume)
        if config.mode != 'fit' or config.resume.model_only:
            # Automatically load model weights in validate/test mode as opposed to using built-in PL argument to
            # validate or test methods since need custom logic e.g. to remove non-dataclass args
            model = model.load_from_checkpoint(checkpoint, **(model_cfg if config.resume.clobber_hparams else {}))
    else:
        checkpoint = None

    if logger:
        if os.environ.get("EXP_LOG_DIR", None) is None:
            # Needed because in distributed training, the logger is not properly initializated on clone processes
            # If dirname for the checkpointer is not the same on all processes, training hangs
            # See https://github.com/PyTorchLightning/pytorch-lightning/issues/5319
            os.environ["EXP_LOG_DIR"] = logger.experiment.dir

    callbacks = []
    checkpoint_callback = 'checkpoint' in config and config.checkpoint is not None

    if logger and checkpoint_callback:
        checkpoint_cb = ModelCheckpoint(**config.checkpoint,
                                        dirpath=Path(os.environ["EXP_LOG_DIR"]) / 'checkpoints')
        checkpoint_cb.CHECKPOINT_NAME_LAST = checkpoint_cb.CHECKPOINT_JOIN_CHAR.join(["{epoch}", "{step}", "last"])
        callbacks.append(checkpoint_cb)

    trainer = pl.Trainer(
        resume_from_checkpoint=None if config.resume.model_only else checkpoint,
        logger=logger,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback,
        **config.trainer
    )

    if config.mode == 'fit':
        trainer.fit(model, datamodule=datamodule)
    elif config.mode == 'validate':
        trainer.validate(model, datamodule=datamodule)
    elif config.mode == 'test':
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    run()
