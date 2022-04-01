import importlib


def get_dataset(cfg):
    module = importlib.import_module('dataset.' + cfg['dataset']['dataset_name'])
    return module.Dataset
