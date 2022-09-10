# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch.utils import data

from lib.config import config
from lib.data import samplers, datasets, collate
from lib.utils.imports import import_file


def setup_dataloader(dataset_name, is_train=True, start_iteration=0, is_iteration_based=True, shuffle=True,
                     pin_memory=True) -> data.DataLoader:
    dataset = build_dataset(dataset_name)

    sampler = samplers.make_data_sampler(dataset, shuffle)

    if is_train:
        iteration_based = is_iteration_based
    else:
        iteration_based = False

    batch_size = config.DATALOADER.IMS_PER_BATCH if is_train else 1
    batch_sampler = samplers.make_batch_data_sampler(sampler, batch_size, config.DATALOADER.MAX_ITER, start_iteration,
                                                     iteration_based=iteration_based)
    collator = collate.BatchCollator()

    data_loader = data.DataLoader(
        dataset,
        num_workers=config.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=pin_memory
    )

    return data_loader


def build_dataset(dataset_name) -> data.Dataset:
    paths_catalog = import_file("lib.config.paths_catalog", config.PATHS_CATALOG, True)
    dataset_catalog = paths_catalog.DatasetCatalog

    info = dataset_catalog.get(dataset_name)
    factory = getattr(datasets, info.pop("factory"))
    info["fields"] = config.DATASETS.FIELDS

    # make dataset from factory
    dataset = factory(**info)

    return dataset
