import copy
from collections import OrderedDict
from pathlib import Path

import torch


def main():
    # load checkpoint
    checkpoint = torch.load("data/panoptic_front3d.pth")
    model_checkpoint = checkpoint["model"]

    # define mappings
    mapping = {
        "proxy_output": "proxy_occupancy_128_head",
        "proxy_instances_128_head": "proxy_instance_128_head",
        "proxy_conv1": "proxy_occupancy_64_head",
        "proxy_semantics": "proxy_semantic_64_head",
        "proxy_instances": "proxy_instance_64_head",
        "completion_256_head": "occupancy_256_head",
        "completion_head": "surface_head"
    }

    # rename parameters according to mapping
    keys_to_update = OrderedDict()

    # search for keys which needs to be updated
    for old, new in mapping.items():
        for group in model_checkpoint:
            if old in group.split("."):
                new_key = group.replace(old, new)
                print(f"{group} -> {new_key}")
                keys_to_update[group] = new_key
                # new_checkpoint[new_key] = model_checkpoint[group]

    # update keys
    new_checkpoint = copy.deepcopy(model_checkpoint)
    for old, new in keys_to_update.items():
        new_checkpoint[new] = new_checkpoint.pop(old)

    print(f"Updated {len(keys_to_update)} keys")

    # save checkpoint
    torch.save({"model": new_checkpoint}, "data/panoptic_front3d_v2.pth")


if __name__ == '__main__':
    main()
