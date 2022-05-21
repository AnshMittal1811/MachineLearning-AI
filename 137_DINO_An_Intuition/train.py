import torch
from losses import DINOLoss

from opt import get_opts

# dataset
from dataset import ImageDataset
from aug_utils import DataAugmentationDINO
from torch.utils.data import DataLoader

# model
import timm
from models import MultiCropWrapper, DINOHead
from losses import DINOLoss

# optimizer
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
from misc import get_learning_rate

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


class DINOSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        arch = f'{hparams.arch}_patch16_224'
        student_backbone = \
            timm.create_model(arch, pretrained=False,
                              drop_path_rate=hparams.drop_path_rate)
        teacher_backbone = timm.create_model(arch, pretrained=False)

        student_head = DINOHead(student_backbone.embed_dim,
                                hparams.out_dim,
                                hparams.norm_last_layer)
        teacher_head = DINOHead(teacher_backbone.embed_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss = DINOLoss(hparams.out_dim,
                             hparams.local_crops_number+2,
                             hparams.warmup_teacher_temp,
                             hparams.teacher_temp,
                             hparams.warmup_teacher_temp_epochs,
                             hparams.num_epochs)

    def setup(self, stage=None):
        transform = DataAugmentationDINO(hparams.global_crops_scale,
                                         hparams.local_crops_scale,
                                         hparams.local_crops_number)
        self.train_dataset = ImageDataset(hparams.root_dir, transform=transform)
        self.val_dataset = ImageDataset(hparams.root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = AdamW(self.student.parameters(),
                               hparams.lr,
                               weight_decay=hparams.weight_decay)
        
        scheduler = CosineLRScheduler(self.optimizer,
                                      t_initial=hparams.num_epochs,
                                      lr_min=hparams.lr/1e2,
                                      warmup_t=hparams.warmup_epochs,
                                      warmup_lr_init=1e-6,
                                      warmup_prefix=True)

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        batch: a list of "2+local_crops_number" tensors
               each tensor is of shape (B, 3, h, w)
        """
        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)

        loss = self.loss(student_output, teacher_output, self.current_epoch)

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it] # momentum parameter
            for ps, pt in \
                zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.mul_(m).add_((1-m)*ps.detach().data)

        self.los('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pass
        # images, labels = batch
        # logits_predicted = self(images)

        # loss = F.cross_entropy(logits_predicted, labels)
        # acc = torch.sum(torch.eq(torch.argmax(logits_predicted, -1), labels).to(torch.float32)) / len(labels)

        # log = {'val_loss': loss,
        #        'val_acc': acc}

        # return log

    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

    #     self.log('val/loss', mean_loss, prog_bar=True)
    #     self.log('val/acc', mean_acc, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    mnistsystem = DINOSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      gradient_clip_val=1.0,
                      precision='bf16' if hparams.use_bf16 else 32,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True)

    trainer.fit(mnistsystem)