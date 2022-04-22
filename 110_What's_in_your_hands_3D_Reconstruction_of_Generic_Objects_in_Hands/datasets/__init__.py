from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from nnutils.mesh_utils import collate_meshes


def build_dataset(args, name, cls, split, is_train=True, **kwarg):
    print(name)
    if name.startswith('obman'):
        from .obman import Obman as dset
    elif name.startswith('ho3d'):
        from .ho3d import HO3D as dset
    elif name.startswith('mow'):
        from .mow import MOW as dset
    else:
        raise NotImplementedError('not implemented %s' % name)
    dset = dset(args, name, split, is_train, data_dir=args.DB.DIR)

    from .sdf_img import SdfImg
    dset = SdfImg(args, dset, is_train, **kwarg)

    dset.preload_anno()
    print(name, len(dset))
    return dset


def build_dataloader(cfg, split, is_train=True, shuffle=None, name=None, cls=None, bs=None) -> DataLoader:
    if shuffle is None:
        shuffle = is_train
    if bs is None:
        bs = cfg.MODEL.BATCH_SIZE if is_train else min(8, cfg.MODEL.BATCH_SIZE)

    if name is None:
        name = cfg.DB.NAME
    if cls is None:
        cls = cfg.DB.CLS
    
    dataset, total_len = [], 0
    for each_name in name.split('+'):
        dset = build_dataset(cfg, each_name, cls, split, is_train, base_idx=total_len)
        total_len += len(dset)
        dataset.append(dset)
        # dataset = [build_dataset(cfg, each_name, cls, split, is_train) for each_name in name.split('+')]
    if '+' in name:
        dataset = ConcatDataset(dataset)
    else:
        dataset = dataset[0]
    print('len', len(dataset), bs)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collate_meshes,
                        shuffle=shuffle, drop_last=is_train, num_workers=10) 
    return loader

