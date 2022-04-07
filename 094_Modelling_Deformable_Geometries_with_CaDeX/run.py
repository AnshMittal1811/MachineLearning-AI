"""
May the Force be with you.
Main program 2019.3
Update 2020.10
Update 2021.7
"""

from dataset import get_dataset
from logger import Logger
from core.models import get_model
from core import runner_dict
from init import get_cfg, setup_seed

cfg = get_cfg()

setup_seed(cfg["rand_seed"])

DatasetClass = get_dataset(cfg)
datasets_dict = dict()
for mode in cfg["modes"]:
    datasets_dict[mode] = DatasetClass(cfg, mode=mode)

ModelClass = get_model(cfg["model"]["model_name"])
model = ModelClass(cfg)

logger = Logger(cfg)

runner = runner_dict[cfg["runner"].lower()](cfg, model, datasets_dict, logger)

runner.run()
