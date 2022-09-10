import yaml
import numpy as np
import collections

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load(path):
    with open('config/default_values.yaml', 'r') as f:
        default_cfg = yaml.load(f, yaml.FullLoader)

    with open(path, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    update(default_cfg, cfg)
    cfg = default_cfg

    cfg['data_bounding_box'] = np.array(cfg['data_bounding_box'])
    cfg['data_bounding_box_str'] = ",".join(str(x) for x in cfg['data_bounding_box'])


    return cfg