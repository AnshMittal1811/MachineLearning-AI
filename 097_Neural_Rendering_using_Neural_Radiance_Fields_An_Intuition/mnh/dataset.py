import os
from .dataset_replica import ReplicaDataset
from .dataset_tat import TanksAndTemplesDataset

def load_datasets(
    dataset_path,
    cfg
):
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path  = os.path.join(dataset_path, 'test')
    train_dataset, valid_dataset = None, None
    
    if 'replica' in cfg.data.path:
        train_dataset = ReplicaDataset(
            folder=train_path, 
            read_points=True, 
            batch_points=cfg.data.batch_points
        )
        valid_dataset = ReplicaDataset(
            folder=valid_path
        )
    elif 'Tanks' in cfg.data.path or 'BlendedMVS' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=valid_path,
        )
    elif 'Synthetic' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=test_path,
        )
    return train_dataset, valid_dataset