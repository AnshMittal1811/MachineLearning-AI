from .logger import Logger


def get_logger(cfg):
    return Logger(cfg)
