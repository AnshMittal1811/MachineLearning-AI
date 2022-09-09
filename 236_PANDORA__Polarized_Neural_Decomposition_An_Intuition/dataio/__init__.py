def get_data(args, return_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get('type', 'DTU')
    cfgs = {
        'scale_radius': args.data.get('scale_radius', -1),
        'downscale': args.data.downscale,
        'data_dir': args.data.data_dir,
        'train_cameras': False
    }
    
    if dataset_type == 'DTU':
        from .DTU import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == 'custom':
        from .custom import SceneDataset
    elif dataset_type == 'BlendedMVS':
        from .BlendedMVS import SceneDataset
    elif dataset_type == 'Mitsuba2':
        from .Mitsuba2 import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
        sel_idx_sample = args.data.get('view_sample',1)
        if sel_idx_sample > 1:
            cfgs['sel_idx_sample'] = sel_idx_sample
            cfgs['sel_idx_offset'] = 0
    elif dataset_type == 'PMVIR':
        from .PMVIR import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == 'Ours':
        from .Ours import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == 'Ours_modmask':
        from .Ours_modmask import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    dataset = SceneDataset(**cfgs)
    if return_val:
        cfgs['downscale'] = val_downscale
        if dataset_type == 'Mitsuba2':
            if sel_idx_sample > 1:
                cfgs['sel_idx_offset'] = 1
        val_dataset = SceneDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset