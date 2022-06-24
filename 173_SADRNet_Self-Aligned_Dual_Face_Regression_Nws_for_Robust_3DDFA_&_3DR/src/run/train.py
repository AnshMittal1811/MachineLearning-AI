import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import config
from src.configs.config_SADRN_v2 import *
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataset.dataloader import make_data_loader, make_dataset
from src.util.printer import DecayVarPrinter


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class BaseTrainer:
    def __init__(self):
        now_time = time.localtime()
        save_dir_time = f"{now_time.tm_year}-{now_time.tm_mon}-{now_time.tm_mday}-{now_time.tm_hour}-{now_time.tm_min}-{now_time.tm_sec}"

        self.model_save_path = f'{config.MODEL_SAVE_DIR}/{config.NET}-{save_dir_time}'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.tb_writer = SummaryWriter(log_dir=self.model_save_path + '/tb')
        self.model = self.get_model()
        self.printer = DecayVarPrinter()

    def get_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class SADRNTrainer(BaseTrainer):
    def __init__(self):
        super(SADRNTrainer, self).__init__()
        self.loss_keys = {'face_uvm', 'kpt_uvm', 'offset_uvm', 'attention_mask', 'smooth'}
        self.metrics_keys = {'face_uvm', 'kpt_uvm', 'offset_uvm', 'attention_mask'}

    def get_model(self):
        from src.model.SADRN import get_model
        return get_model()

    def load_weight(self, model, weight_path):
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        match_dict = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(match_dict)
        model.load_state_dict(model_dict)
        model = model.to(config.DEVICE)
        return model

    def train(self):
        model = self.model
        model = model.to(config.DEVICE)
        optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
        scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=3, after_scheduler=scheduler_exp)

        scheduler = scheduler_warmup

        train_dataset = make_dataset(config.TRAIN_DIR, 'train')
        val_dataset = make_dataset(config.VAL_DIR, 'val')
        train_data_loader = make_data_loader(train_dataset.train_data, mode=config.DATA_TYPE,
                                             batch_size=config.BATCH_SIZE, is_shuffle=True, is_aug=True,
                                             num_worker=config.DATALOADER_WORKER_NUM)
        val_data_loader = make_data_loader(val_dataset.val_data, mode=config.DATA_TYPE,
                                           batch_size=config.BATCH_SIZE, is_shuffle=False, is_aug=False,
                                           num_worker=config.DATALOADER_WORKER_NUM)

        best_val_error = 1e5

        for _ in range(config.START_EPOCH):
            scheduler.step()
        loss_keys = self.loss_keys
        metrics_keys = self.metrics_keys

        for epoch in range(config.START_EPOCH, config.TOTAL_EPOCH):
            print('Epoch: %d' % epoch)
            self.printer.clear()
            model.train()
            total_itr_num = len(train_data_loader.dataset) // train_data_loader.batch_size

            t_start = time.time()
            for i, (img, gt_face_uvm, gt_offset, gt_attention) in enumerate(train_data_loader):
                optimizer.zero_grad()

                # 准备数据
                img = img.to(config.DEVICE).float()
                gt_face_uvm = gt_face_uvm.to(config.DEVICE).float()
                gt_offset = gt_offset.to(config.DEVICE).float()
                gt_attention = gt_attention.to(config.DEVICE).float()

                loss = model({'img': img}, {'face_uvm': gt_face_uvm, 'attention_mask': gt_attention, 'offset_uvm': gt_offset}, 'train')
                loss = {k: loss[k].mean() for k in loss}
                # backward
                sum_loss = sum(loss[k] for k in loss)
                sum_loss.backward()
                optimizer.step()

                self.printer.update_variable_decay('loss', sum_loss)
                for k in loss.keys():
                    self.printer.update_variable_decay(k, loss[k])

                print('\r', end='')
                print('[epoch:%d, iter:%d/%d, time:%d] %s ' % (
                    epoch, i + 1, total_itr_num, int(time.time() - t_start), self.printer.get_variable_str('loss')),
                      end='')
                for k in loss.keys():
                    print(self.printer.get_variable_str(k), end=' ')

            # validation

            with torch.no_grad():
                model.eval()
                val_i = 0
                print("\nWaiting Test!", val_i, end='\r')
                for i, (img, gt_face_uvm, gt_offset, gt_attention) in enumerate(val_data_loader):
                    val_i += 1
                    print("Waiting Test!", val_i, end='\r')
                    img = img.to(config.DEVICE).float()
                    gt_face_uvm = gt_face_uvm.to(config.DEVICE).float()
                    gt_offset = gt_offset.to(config.DEVICE).float()
                    gt_attention = gt_attention.to(config.DEVICE).float()

                    error = model({'img': img}, {'face_uvm': gt_face_uvm, 'attention_mask': gt_attention, 'offset_uvm': gt_offset}, 'eval')
                    error = {k: error[k].mean() for k in error}
                    # backward
                    sum_error = sum(error[k] for k in error)
                    self.printer.update_variable_avg('val_error', sum_error)
                    for k in error.keys():
                        self.printer.update_variable_avg('val_' + k, error[k])

                    for k in error.keys():
                        print(self.printer.get_variable_str('val_' + k), end=' ')

                val_error = self.printer.get_variable_val('val_error')

                print('\nSaving model......', end='\r')
                torch.save(model.state_dict(), '%s/net_%03d.pth' % (self.model_save_path, epoch))
                # save
                if val_error < best_val_error:
                    print('new best %.4f improved from %.4f' % (val_error, best_val_error))
                    best_val_error = val_error
                    torch.save(model.state_dict(), '%s/best.pth' % self.model_save_path)
                else:
                    print('not improved from %.4f' % best_val_error)

            # write log
            self.tb_writer.add_scalar('train/loss', self.printer.get_variable_val('loss'), epoch)
            for k in loss_keys:
                self.tb_writer.add_scalar('train/%s' % k, self.printer.get_variable_val(k), epoch)
            for k in metrics_keys:
                self.tb_writer.add_scalar('val/%s' % k, self.printer.get_variable_val('val_' + k), epoch)

            scheduler.step()
        self.tb_writer.close()


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='model arguments')
    parser.add_argument('--visible_device', default='0', type=str, help='')
    run_args = parser.parse_args()
    print(run_args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = run_args.visible_device
    print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(),
          torch.cuda.get_device_name(0))

    if config.NET == 'SADRN':
        trainer = SADRNTrainer()
        trainer.train()

