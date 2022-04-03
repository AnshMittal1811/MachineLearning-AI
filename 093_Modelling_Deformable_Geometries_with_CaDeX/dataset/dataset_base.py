"""
Base class for dataset
"""
import torch.utils.data as data
from copy import deepcopy
import os
import logging
import torch
from multiprocessing.dummy import Pool


class DatasetBase(data.Dataset):
    def __init__(self, cfg, mode):
        """
        Dataset Base

        Method to implement:
        1. def __prepare_meta_list__(self, cfg)
            use init to find the index of dataset
        2. def __read_into_ram__(self, meta_info)
            load one data-point to the memory
        3. def __prepare_from_ram__(self, data)
            some operations before send raw data from memory to network

        Parameters
        ----------
        mode: each dataset must be specified with one mode
        """

        self.cfg = deepcopy(cfg)
        self.mode = mode
        self.dataset_root = os.path.join(self.cfg["root"], self.cfg["dataset"]["dataset_root"])

        self.dataset_proportion = self.cfg["dataset"]["dataset_proportion"][
            self.cfg["modes"].index(self.mode)
        ]
        self.meta_info_list = self.__prepare_meta_list__(self.cfg)
        if self.dataset_proportion < 1:
            self.meta_info_list = self.meta_info_list[
                : int(len(self.meta_info_list) * self.dataset_proportion)
            ]
        # cache dataset
        self.cache_flag = self.cfg["dataset"]["ram_cache"]
        self.ram_cache_list = []
        if self.cache_flag:
            # self.__cache_dataset__()
            self.__cache_dataset_parallel__()

        logging.debug(
            "Initialized a size {} {} dataset".format(
                self.__len__(), self.cfg["dataset"]["dataset_name"]
            )
        )
        logging.debug("With {} mode on dataset root: {}".format(self.mode, self.dataset_root))

        return

    def __cache_dataset_thread__(self, ind, meta):
        ret = None
        try:
            self.ram_cache_list.append(self.__read_into_ram__(meta))
        except:
            logging.warning("Data sample {} read fail, omit this data point".format(ind))
            ret = ind
        if len(self.ram_cache_list) % 100 == 0:
            logging.debug(
                "Cached {}/{} datapoints".format(len(self.ram_cache_list), self.__len__())
            )
        return ret

    def __cache_dataset_parallel__(self):
        # make task
        tasks = [(ind, meta) for ind, meta in enumerate(self.meta_info_list)]
        # cache
        k = self.cfg["dataset"]["num_workers"]
        logging.info("Caching dataset with {} threads ...".format(k))
        with Pool(k) as pool:
            fail_ind = [pool.apply_async(self.__cache_dataset_thread__, t) for t in tasks]
            fail_ind = [i.get() for i in fail_ind]
        fail_ind = [i for i in fail_ind if i is not None]
        fail_ind.sort()
        for ind in fail_ind[::-1]:
            self.meta_info_list.pop(ind)
        return

    def __cache_dataset__(self):
        # single thread, old version of caching
        read_fail_ind_list = []
        logging.info("Caching Dataset ... ")
        for ind, meta in enumerate(self.meta_info_list):
            if ind % 100 == 0:
                logging.debug("Cached {}/{} datapoints".format(ind, self.__len__()))
            try:
                self.ram_cache_list.append(self.__read_into_ram__(meta))
            except:
                logging.warning("Data sample {} read fail, omit this data point".format(ind))
                read_fail_ind_list.append(ind)
        for ind in read_fail_ind_list[::-1]:
            self.meta_info_list.pop(ind)
        if len(read_fail_ind_list) > 0:
            logging.warning(
                "Warnning! There are {} damaged datapoint, removed "
                "from dataset".format(len(read_fail_ind_list))
            )
        logging.info("Caching finished")

    def __prepare_meta_list__(self, cfg):
        """
        Need Implementation
        prepare the core of the dataset
        """
        return [None]

    def __read_into_ram__(self, meta_info):
        """
        Need Implementation
        load one data sample into memory
        """
        data_from_disk = None
        return data_from_disk

    def __prepare_from_ram__(self, data):
        """
        Need Implementation
        From cache list or raw data read from disk, pack them to the output of dataset (input of networks)
        """
        return None

    def __len__(self):
        return len(self.meta_info_list)

    def __getitem__(self, item):
        meta_info = self.meta_info_list[item]
        if self.cache_flag:
            raw_data = self.ram_cache_list[item]
        else:
            raw_data = self.__read_into_ram__(meta_info)
        data = self.__prepare_from_ram__(raw_data)
        data["dataset_ind"] = item
        assert "viz_id" in meta_info.keys()
        return data, meta_info
