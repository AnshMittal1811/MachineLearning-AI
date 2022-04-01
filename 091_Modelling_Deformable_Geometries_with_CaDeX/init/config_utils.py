import yaml
import logging
import os


def load_config(path, default_path=None):
    ''' Loads init file.
    from https://github.com/autonomousvision/occupancy_flow

    Args:
        path (str): path to init file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # Check if we should inherit from a init
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this init first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # Include main configuration
    cfg = update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two init dictionaries recursively.
    from https://github.com/autonomousvision/occupancy_flow

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def configure_logging(cfg, time_stamp):
    """
    https://github.com/facebookresearch/DeepSDF
    """
    logger = logging.getLogger()
    if cfg['logging']['debug_mode']:
        logger.setLevel(logging.DEBUG)
    # elif args.quiet:
    #     logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("| " + cfg['method'] + " | %(levelname)s | %(asctime)s | %(message)s",
                                  "%Y%b%d-%H:%M:%S")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    log_path = os.path.join(cfg['root'], 'log', cfg['logging']['log_dir'], 'runtime_cmd_log_files') 
    os.makedirs(log_path, exist_ok=True)
    file_logger_handler = logging.FileHandler(
        os.path.join(log_path, 'running_log_start_time_{}.log'.format(time_stamp)))
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)
