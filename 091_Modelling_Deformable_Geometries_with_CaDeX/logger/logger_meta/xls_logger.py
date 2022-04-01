"""
excel logger
data structure:
- one head-key is one file
- each passed in data is a dict {col-name:list of values}, each value will be recorded into one row
- there is some basic meta info for each row
"""

import pandas as pd
from .base_logger import BaseLogger
import os
import logging


class XLSLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "xls"
        os.makedirs(self.log_path, exist_ok=True)
        self.pd_container = dict()

        self.current_epoch = 1
        self.current_phase = "INIT"

    def log_batch(self, batch):
        # get data
        if self.NAME not in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch["data"]
        phase = batch["phase"]
        current_epoch = batch["epoch"]
        self.current_epoch = current_epoch
        self.current_phase = phase
        meta_info = batch["meta_info"]
        # for each key (file)
        for sheet_key in keys_list:
            if sheet_key not in data.keys():
                continue
            kdata = data[sheet_key]
            assert isinstance(kdata, dict)
            if sheet_key not in self.pd_container.keys():
                self.pd_container[sheet_key] = pd.DataFrame()
            add_list = list()
            count = len(meta_info["viz_id"])
            for ii in range(count):
                _data = dict()
                for k, v in kdata.items():
                    _data[k] = v[ii]
                _data["viz_id"] = meta_info["viz_id"][ii]
                add_list.append(_data)
            self.pd_container[sheet_key] = self.pd_container[sheet_key].append(
                add_list, ignore_index=True
            )

    def log_phase(self):
        for k in self.pd_container.keys():
            # handle end log
            if len(self.pd_container[k]) == 0:
                continue
            try:
                D = self.pd_container[k]
                df2 = pd.DataFrame(D.mean(axis=0))
                self.pd_container[k] = pd.concat([df2.T, D], axis=0, ignore_index=False)
            except:
                logging.warning("XLS loger add mean to head fail, ignore and continue")
            self.pd_container[k].to_excel(
                os.path.join(
                    self.log_path,
                    k + "_" + str(self.current_epoch) + "_" + self.current_phase + ".xls",
                )
            )
            self.pd_container[k] = pd.DataFrame()
