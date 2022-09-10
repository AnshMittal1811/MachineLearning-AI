import time
from collections import OrderedDict
from pathlib import Path

import torch
from lib.structures.field_list import collect

from lib import utils, logger, config, modeling, solver, data


class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpointer = None
        self.dataloader = None
        self.logger = logger
        self.meters = utils.MetricLogger(delimiter="  ")
        self.checkpoint_arguments = {}

        self.setup()

    def setup(self) -> None:
        # Setup model
        self.model = modeling.PanopticReconstruction()

        device = torch.device(config.MODEL.DEVICE)
        self.model.to(device, non_blocking=True)

        self.model.log_model_info()
        self.model.fix_weights()

        # Setup optimizer, scheduler, checkpointer
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
        self.scheduler = solver.WarmupMultiStepLR(self.optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA,
                                                  warmup_factor=1,
                                                  warmup_iters=0,
                                                  warmup_method="linear")

        output_path = Path(config.OUTPUT_DIR)
        self.checkpointer = utils.DetectronCheckpointer(self.model, self.optimizer, self.scheduler, output_path)

        # Load the checkpoint
        checkpoint_data = self.checkpointer.load()

        # Additionally load a 2D model which overwrites the previously loaded weights
        # TODO: move to checkpointer?
        if config.MODEL.PRETRAIN2D:
            pretrain_2d = torch.load(config.MODEL.PRETRAIN2D)
            self.model.load_state_dict(pretrain_2d["model"])

        self.checkpoint_arguments["iteration"] = 0

        if config.SOLVER.LOAD_SCHEDULER:
            self.checkpoint_arguments.update(checkpoint_data)

        # Dataloader
        self.dataloader = data.setup_dataloader(config.DATASETS.TRAIN)

    def do_train(self) -> None:
        # Log start logging
        self.logger.info(f"Start training {self.checkpointer.output_path.name}")

        # Switch training mode
        self.model.switch_training()

        # Main loop
        iteration = 0
        iteration_end = time.time()

        for idx, (image_ids, targets) in enumerate(self.dataloader):
            assert targets is not None, "error during data loading"
            data_time = time.time() - iteration_end

            # Get input images
            images = collect(targets, "color")

            # Pass through model
            try:
                losses, results = self.model(images, targets)
            except Exception as e:
                print(e, "skipping", image_ids[0])
                del targets, images
                continue

            # Accumulate total loss
            total_loss: torch.Tensor = 0.0
            log_meters = OrderedDict()

            for loss_group in losses.values():
                for loss_name, loss in loss_group.items():
                    if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss
                        log_meters[loss_name] = loss.item()

            # Loss backpropagation, optimizer & scheduler step
            self.optimizer.zero_grad()

            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log_meters["total"] = total_loss.item()
            else:
                log_meters["total"] = total_loss

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()

            # Save checkpoint
            if iteration % config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.checkpointer.save(f"model_{iteration:07d}", **self.checkpoint_arguments)

            last_training_stage = self.model.set_current_training_stage(iteration)

            # Save additional checkpoint after hierarchy level
            if last_training_stage is not None:
                self.checkpointer.save(f"model_{last_training_stage}_{iteration:07d}", **self.checkpoint_arguments)
                self.logger.info(f"Finish {last_training_stage} hierarchy level")

            # Gather logging information
            self.meters.update(**log_meters)
            batch_time = time.time() - iteration_end
            self.meters.update(time=batch_time, data=data_time)
            current_learning_rate = self.scheduler.get_lr()[0]
            current_training_stage = self.model.get_current_training_stage()

            self.logger.info(self.meters.delimiter.join([f"IT: {iteration:06d}", current_training_stage,
                                                         f"{str(self.meters)}", f"LR: {current_learning_rate}"]))

            iteration += 1
            iteration_end = time.time()

        self.checkpointer.save("model_final", **self.checkpoint_arguments)
