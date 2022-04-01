# from https://github.com/autonomousvision/occupancy_flow
from torchvision import transforms
import dataset.oflow_dataset as oflow_dataset
from torch.utils.data import Dataset
import logging


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        self.dataset = get_dataset(mode, cfg)
        self.mode = mode.lower()
        self.dataset_proportion = cfg["dataset"]["dataset_proportion"][
            cfg["modes"].index(self.mode)
        ]
        # ! add for save render results
        if "one_per_instance" in cfg["dataset"].keys():
            if cfg["dataset"]["one_per_instance"]:
                start_idx = 50
                filtered_list = []
                for dp in self.dataset.models:
                    if dp["start_idx"] == start_idx:
                        filtered_list.append(dp)
                self.dataset.models = filtered_list
        self.dataset.models = self.dataset.models[
            : int(self.dataset_proportion * len(self.dataset.models))
        ]
        logging.info(
            "Use dataset implemented by O-Flow: https://github.com/autonomousvision/occupancy_flow"
        )
        logging.info(
            "Dataset {} with {}% data, dataset size is {}".format(
                mode, self.dataset_proportion * 100, len(self.dataset)
            )
        )
        seq_len_train = cfg["dataset"]["oflow_config"]["length_sequence"]
        if "length_sequence_val" in cfg["dataset"]["oflow_config"].keys():
            seq_len_val = cfg["dataset"]["oflow_config"]["length_sequence_val"]
        else:
            seq_len_val = seq_len_train
        self.seq_length = seq_len_train if mode == "train" else seq_len_val
        if "n_training_frames" in cfg["dataset"].keys():
            self.n_training_frames = cfg["dataset"]["n_training_frames"]
        else:
            self.n_training_frames = -1

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        meta_info = self.dataset.models[index]
        viz_id = "{}_".format(index)
        for v in meta_info.values():
            viz_id += str(v) + "_"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode
        if "points" in data.keys():
            if data["points"].ndim == 3:
                try:
                    if self.n_training_frames > 0 and self.mode == "train":
                        assert data["points"].shape[0] == self.n_training_frames
                    else:
                        assert data["points"].shape[0] == self.seq_length
                except:
                    print(data["points"].shape[0])
                    raise RuntimeError("Data Length Invalid")
        return data, meta_info


def get_dataset(mode, cfg, return_idx=False, return_category=False):
    """Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    method = cfg["method"]
    dataset_type = cfg["dataset"]["oflow_config"]["dataset"]
    dataset_folder = cfg["dataset"]["oflow_config"]["path"]
    categories = cfg["dataset"]["oflow_config"]["classes"]

    # Get split
    splits = {
        "train": cfg["dataset"]["oflow_config"]["train_split"],
        "val": cfg["dataset"]["oflow_config"]["val_split"],
        "test": cfg["dataset"]["oflow_config"]["test_split"],
    }
    split = splits[mode]
    # Create dataset
    if dataset_type == "Humans":
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields["inputs"] = inputs_field

        if return_idx:
            fields["idx"] = oflow_dataset.IndexField()

        if return_category:
            fields["category"] = oflow_dataset.CategoryField()

        seq_len_train = cfg["dataset"]["oflow_config"]["length_sequence"]
        if "length_sequence_val" in cfg["dataset"]["oflow_config"].keys():
            seq_len_val = cfg["dataset"]["oflow_config"]["length_sequence_val"]
        else:
            seq_len_val = seq_len_train
        if mode == "train":
            seq_len = seq_len_train
        else:
            seq_len = seq_len_val

        dataset = oflow_dataset.HumansDataset(
            dataset_folder,
            fields,
            split=split,
            categories=categories,
            length_sequence=seq_len,
            n_files_per_sequence=cfg["dataset"]["oflow_config"]["n_files_per_sequence"],
            offset_sequence=cfg["dataset"]["oflow_config"]["offset_sequence"],
            ex_folder_name=cfg["dataset"]["oflow_config"]["pointcloud_seq_folder"],
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg["dataset"]["oflow_config"]["dataset"])

    return dataset


def get_data_fields(mode, cfg):
    """Returns data fields.

    Args:
        mode (str): mode (train|val|test)
        cfg (yaml config): yaml config object
    """
    fields = {}
    seq_len_train = cfg["dataset"]["oflow_config"]["length_sequence"]
    if "length_sequence_val" in cfg["dataset"]["oflow_config"].keys():
        seq_len_val = cfg["dataset"]["oflow_config"]["length_sequence_val"]
    else:
        seq_len_val = seq_len_train
    p_folder = cfg["dataset"]["oflow_config"]["points_iou_seq_folder"]
    pcl_folder = cfg["dataset"]["oflow_config"]["pointcloud_seq_folder"]
    mesh_folder = cfg["dataset"]["oflow_config"]["mesh_seq_folder"]
    generate_interpolate = cfg["dataset"]["oflow_config"]["generation_interpolate"]
    unpackbits = cfg["dataset"]["oflow_config"]["points_unpackbits"]
    if "training_all_steps" in cfg["dataset"]["oflow_config"].keys():
        training_all = cfg["dataset"]["oflow_config"]["training_all_steps"]
    else:
        training_all = False
    if "n_training_frames" in cfg["dataset"].keys():
        n_training_frames = cfg["dataset"]["n_training_frames"]
    else:
        n_training_frames = -1

    # Transformation
    transf_pt, transf_pt_val, transf_pcl, transf_pcl_val = get_transforms(cfg)

    # Fields
    pts_iou_field = oflow_dataset.PointsSubseqField
    pts_corr_field = oflow_dataset.PointCloudSubseqField

    if "not_choose_last" in cfg["dataset"].keys():
        not_choose_last = cfg["dataset"]["not_choose_last"]
    else:
        not_choose_last = False
    training_multi_files = False
    if "training_multi_files" in cfg["dataset"]["oflow_config"]:
        if cfg["dataset"]["oflow_config"]["training_multi_files"]:
            training_multi_files = True
            logging.info("Oflow D-FAUST Points Field use multi files to speed up disk performation")

    if mode == "train":
        if cfg["model"]["loss_recon"]:
            if training_all:
                fields["points"] = pts_iou_field(
                    p_folder,
                    transform=transf_pt,
                    all_steps=True,
                    seq_len=seq_len_train,
                    unpackbits=unpackbits,
                    use_multi_files=training_multi_files,
                )
            else:
                fields["points"] = pts_iou_field(
                    p_folder,
                    sample_nframes=n_training_frames,
                    transform=transf_pt,
                    seq_len=seq_len_train,
                    fixed_time_step=0,
                    unpackbits=unpackbits,
                    use_multi_files=training_multi_files,
                )
            fields["points_t"] = pts_iou_field(
                p_folder,
                transform=transf_pt,
                seq_len=seq_len_train,
                unpackbits=unpackbits,
                not_choose_last=not_choose_last,
                use_multi_files=training_multi_files,
            )
    # only training can be boost by multi-files
    # modify here, if not train, val should also load the same as the test
    else:
        fields["points"] = pts_iou_field(
            p_folder,
            transform=transf_pt_val,
            all_steps=True,
            seq_len=seq_len_val,
            unpackbits=unpackbits,
        )
        fields[
            "points_mesh"
        ] = pts_corr_field(  # ? this if for correspondence? Checked, this is for chamfer distance, make sure that because here we use tranforms, teh pts in config file must be 100000
            pcl_folder, transform=transf_pcl_val, seq_len=seq_len_val
        )
    # Connectivity Loss:
    if cfg["model"]["loss_corr"]:
        fields["pointcloud"] = pts_corr_field(
            pcl_folder,
            transform=transf_pcl,
            seq_len=seq_len_train,
            use_multi_files=training_multi_files,
        )
    if mode == "test" and generate_interpolate:
        fields["mesh"] = oflow_dataset.MeshSubseqField(
            mesh_folder, seq_len=seq_len_val, only_end_points=True
        )
    fields["oflow_idx"] = oflow_dataset.IndexField()
    return fields


def get_transforms(cfg):
    """Returns transform objects.

    Args:
        cfg (yaml config): yaml config object
    """
    n_pcl = cfg["dataset"]["oflow_config"]["n_training_pcl_points"]
    n_pt = cfg["dataset"]["oflow_config"]["n_training_points"]
    n_pt_eval = cfg["dataset"]["n_query_sample_eval"]

    transf_pt = oflow_dataset.SubsamplePoints(n_pt)
    transf_pt_val = oflow_dataset.SubsamplePointsSeq(n_pt_eval, random=False)
    transf_pcl_val = oflow_dataset.SubsamplePointcloudSeq(n_pt_eval, random=False)
    transf_pcl = oflow_dataset.SubsamplePointcloudSeq(n_pcl, connected_samples=True)

    return transf_pt, transf_pt_val, transf_pcl, transf_pcl_val


def get_inputs_field(mode, cfg):
    """Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    input_type = cfg["dataset"]["oflow_config"]["input_type"]
    seq_len_train = cfg["dataset"]["oflow_config"]["length_sequence"]
    if "length_sequence_val" in cfg["dataset"]["oflow_config"].keys():
        seq_len_val = cfg["dataset"]["oflow_config"]["length_sequence_val"]
    else:
        seq_len_val = seq_len_train
    if mode == "train":
        seq_len = seq_len_train
    else:
        seq_len = seq_len_val

    if input_type is None:
        inputs_field = None
    elif input_type == "img_seq":
        if mode == "train" and cfg["dataset"]["oflow_config"]["img_augment"]:
            resize_op = transforms.RandomResizedCrop(
                cfg["dataset"]["oflow_config"]["img_size"], (0.75, 1.0), (1.0, 1.0)
            )
        else:
            resize_op = transforms.Resize((cfg["dataset"]["oflow_config"]["img_size"]))

        transform = transforms.Compose(
            [
                resize_op,
                transforms.ToTensor(),
            ]
        )

        if mode == "train":
            random_view = True
        else:
            random_view = False

        inputs_field = oflow_dataset.ImageSubseqField(
            cfg["dataset"]["oflow_config"]["img_seq_folder"], transform, random_view=random_view
        )
    elif input_type == "pcl_seq":
        connected_samples = cfg["dataset"]["oflow_config"]["input_pointcloud_corresponding"]
        transform = transforms.Compose(
            [
                oflow_dataset.SubsamplePointcloudSeq(
                    cfg["dataset"]["oflow_config"]["input_pointcloud_n"],
                    connected_samples=connected_samples,
                ),
                oflow_dataset.PointcloudNoise(
                    cfg["dataset"]["oflow_config"]["input_pointcloud_noise"]
                ),
            ]
        )
        training_multi_files = False
        if "training_multi_files" in cfg["dataset"]["oflow_config"]:
            if cfg["dataset"]["oflow_config"]["training_multi_files"] and mode == "train":
                training_multi_files = True
                logging.info(
                    "Oflow D-FAUST PCL Field use multi files to speed up disk performation"
                )

        inputs_field = oflow_dataset.PointCloudSubseqField(
            cfg["dataset"]["oflow_config"]["pointcloud_seq_folder"],
            transform,
            seq_len=seq_len,
            use_multi_files=training_multi_files,
        )
    elif input_type == "end_pointclouds":
        transform = oflow_dataset.SubsamplePointcloudSeq(
            cfg["dataset"]["oflow_config"]["input_pointcloud_n"],
            connected_samples=cfg["dataset"]["oflow_config"]["input_pointcloud_corresponding"],
        )

        inputs_field = oflow_dataset.PointCloudSubseqField(
            cfg["dataset"]["oflow_config"]["pointcloud_seq_folder"],
            only_end_points=True,
            seq_len=seq_len,
            transform=transform,
        )
    elif input_type == "idx":
        inputs_field = oflow_dataset.IndexField()
    else:
        raise ValueError("Invalid input type (%s)" % input_type)
    return inputs_field
