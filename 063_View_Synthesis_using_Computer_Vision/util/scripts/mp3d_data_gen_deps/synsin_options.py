# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

def get_dataset(opt):

    print("Loading dataset %s ..." % opt.dataset)
    if opt.dataset == "mp3d":
        opt.train_data_path = (
            opt.habitat_api_prefix
            + "habitat-api/data/datasets/pointnav/mp3d/v1/train/train.json.gz"
        )
        opt.val_data_path = (
            opt.habitat_api_prefix
            + "habitat-api/data/datasets/pointnav/mp3d/v1/test/test.json.gz"
        )
        opt.test_data_path = (
            opt.habitat_api_prefix
            + "habitat-api/data/datasets/pointnav/mp3d/v1/val/val.json.gz"
        )
        opt.scenes_dir = opt.habitat_api_prefix + "habitat-api/data/scene_datasets/" # this should store mp3d
        opt.config = opt.habitat_api_prefix + "habitat-api/configs/tasks/pointnav_rgbd.yaml"
    elif opt.dataset == "habitat":
        opt.train_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz"
        )
        opt.val_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
        )
        opt.test_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/habitat-test-scenes/v1/test/test.json.gz"
        )
        opt.scenes_dir = "/private/home/ow045820/projects/habitat/habitat-api//data/scene_datasets"
    elif opt.dataset == "replica":
        opt.train_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/train/train.json.gz"
        )
        opt.val_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/val/val.json.gz"
        )
        opt.test_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/test/test.json.gz"
        )
        opt.scenes_dir = "/checkpoint/ow045820/data/replica/"
    elif opt.dataset == "realestate":
        opt.min_z = 1.0
        opt.max_z = 100.0
        opt.train_data_path = (
            "/checkpoint/ow045820/data/realestate10K/RealEstate10K/"
        )
        from data.realestate10k import RealEstate10K

        return RealEstate10K
    elif opt.dataset == 'kitti':
        opt.min_z = 1.0
        opt.max_z = 50.0
        opt.train_data_path = (
            '/private/home/ow045820/projects/code/continuous_view_synthesis/datasets/dataset_kitti'
        )
        from data.kitti import KITTIDataLoader

        return KITTIDataLoader

    from util.scripts.mp3d_data_gen_deps.synsin_habitat_data import HabitatImageGenerator as Dataset

    return Dataset
