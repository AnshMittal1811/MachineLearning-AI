"""
06/31/2019
Each passed in batch must contain info:
    1. method of model to save itself
    2. the way to parsing the batch keys
    3. meta information of each sample
    4. batch head (phase)
Logger can be selected from init file
"""
import os
from tensorboardX import SummaryWriter as writer
from .logger_meta import LOGGER_REGISTED
from copy import deepcopy
import logging


class Logger(object):

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        tb_path = os.path.join(cfg['root'], 'log', cfg['logging']['log_dir'], 'tensorboardx')
        self.tb_writer = writer(tb_path)
        self.logger_list = self.compose(self.cfg['logging']['loggers'])
        return

    def compose(self, names):
        loggers_list = list()
        mapping = LOGGER_REGISTED
        for name in names:
            if name in mapping.keys():
                loggers_list.append(mapping[name](self.tb_writer, os.path.join(
                    os.path.join(self.cfg['root'], 'log', self.cfg['logging']['log_dir'], name)), self.cfg))
            else:
                raise Warning('Required logger ' + name + ' not found!')
        logging.debug("Loggers [{}] registered".format(names))
        return loggers_list

    def log_phase(self):
        for lgr in self.logger_list:
            lgr.log_phase()

    def log_batch(self, batch):
        for lgr in self.logger_list:
            lgr.log_batch(batch)

    def end_log(self):
        for lgr in self.logger_list:
            lgr.log_phase()
