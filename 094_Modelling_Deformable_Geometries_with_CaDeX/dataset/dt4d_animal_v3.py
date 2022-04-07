# new dataset for dt4d dataset

from torch.utils.data import Dataset
import logging
import json
import os
import numpy as np
from os.path import join


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        # init
        self.mode = mode.lower()
        self.dataset_proportion = cfg["dataset"]["dataset_proportion"][
            cfg["modes"].index(self.mode)
        ]
        self.data_root = join(cfg["root"], cfg["dataset"]["data_root"])

        self.occ_n_chunk = cfg["dataset"]["occ_n_chunk"]
        self.corr_n_chunk = cfg["dataset"]["corr_n_chunk"]
        self.chunk_size = cfg["dataset"]["chunk_size"]

        self.seq_len = cfg["dataset"]["seq_len"]
        self.n_training_frames = cfg["dataset"]["num_training_frames"]
        self.n_inputs = cfg["dataset"]["num_input_pts"]
        self.inputs_noise_std = cfg["dataset"]["input_noise"]
        self.n_uni = cfg["dataset"]["num_query_uni"]
        self.n_nss = cfg["dataset"]["num_query_ns"]
        self.n_corr = cfg["dataset"]["num_corr_pts"]
        self.n_query_eval = cfg["dataset"]["n_query_sample_eval"]

        self.one_per_instance = False
        if "one_per_instance" in cfg["dataset"].keys():
            self.one_per_instance = cfg["dataset"]["one_per_instance"]
        if self.mode != "train":
            logging.info(f"{self.mode} eval_n set to {self.n_query_eval}")
            logging.info("Be careful! Test eval_n should set to 100000")

        self.use_camera_frame = True  # default is true
        if "camera_frame" in cfg["dataset"].keys():
            self.use_camera_frame = cfg["dataset"]["camera_frame"]

        self.oflow_flag = False
        # O-Flow let points to be at t=0 and points_t to be at t during training, during testing, points are for full time
        if "oflow_flag" in cfg["dataset"].keys():
            self.oflow_flag = cfg["dataset"]["oflow_flag"]

        # build meta info
        self.meta_list = []
        self.input_type = cfg["dataset"]["input_type"]
        assert self.input_type in ["scan", "static", "pcl"]
        split_fn = cfg["dataset"]["split"][self.mode]
        if "sub_cate" in cfg["dataset"].keys():
            if len(cfg["dataset"]["sub_cate"]) > 0:
                split_fn = f"{cfg['dataset']['sub_cate']}_" + split_fn
                logging.info(f"Use {cfg['dataset']['sub_cate']} dataset split {split_fn}")
        with open(join(self.data_root, "index", split_fn)) as f:
            split_data = json.load(f)
        if self.input_type == "pcl":  # spare point-cloud input
            for dp in split_data:
                seq_dir = join(self.data_root, dp)
                T = int(len(os.listdir(join(seq_dir, "c_occ"))) / self.occ_n_chunk)
                for t in range(T):
                    if t + self.seq_len <= T:
                        self.meta_list.append({"seq_dir": seq_dir, "start": t})
        else:  # static or circle scan view
            self.views_use = cfg["dataset"]["num_view"]
            view_sub_folders = [f"{self.input_type}_{i}" for i in range(self.views_use)]
            self.withhold_views = (
                cfg["dataset"]["withhold_view"] and self.views_use >= 2
            )  # only valid when we use 2 views
            if self.withhold_views:  # withhold some views for UV test
                wv_split_fn = cfg["dataset"]["split"]["withhold_view"]
                if "sub_cate" in cfg["dataset"].keys():
                    if len(cfg["dataset"]["sub_cate"]) > 0:
                        wv_split_fn = f"{cfg['dataset']['sub_cate']}_" + wv_split_fn
                with open(join(self.data_root, "index", wv_split_fn)) as f:
                    wv_split_data = json.load(f)
            else:
                wv_split_data = []
            for dp in split_data:
                for vi, view_sub in enumerate(view_sub_folders):
                    if vi == 0 and dp in wv_split_data and self.mode == "train":
                        continue  # for training, skip some withholding views
                    seq_dir = join(self.data_root, dp)
                    view_dir = join(seq_dir, view_sub)
                    T = len(os.listdir(join(view_dir)))
                    for t in range(T):
                        if t + self.seq_len <= T:
                            self.meta_list.append(
                                {"seq_dir": seq_dir, "view": view_sub, "start": t}
                            )
        # for overfitting development, id filter
        if "id_filter_list" in cfg["dataset"].keys():
            _filtered_list = []
            for meta in self.meta_list:
                for v in cfg["dataset"]["id_filter_list"]:
                    if v in meta["seq_dir"]:
                        _filtered_list.append(meta)
                        break
            self.meta_list = _filtered_list

        # ! For save render meshes
        if self.one_per_instance:
            filtered_list = []
            if self.input_type == "pcl":
                start = 0  # 17
                for dp in self.meta_list:
                    if dp["start"] == start:
                        filtered_list.append(dp)
            else:
                start = 0  # 17
                view = "0"
                for dp in self.meta_list:
                    if dp["view"].endswith(view) and dp["start"] == start:
                        filtered_list.append(dp)
            self.meta_list = filtered_list
        self.meta_list = self.meta_list[: int(len(self.meta_list) * self.dataset_proportion)]

        logging.info(
            "Dataset {} with {}% data, dataset size is {}".format(
                mode, self.dataset_proportion * 100, len(self.meta_list)
            )
        )
        return

    def __len__(self) -> int:
        return len(self.meta_list)

    def get_chunk_index(self, n, type, random_flag):
        assert type in ["occ", "corr"]
        n_chunk_file = int(np.ceil(n / float(self.chunk_size)))
        max_chunk = self.corr_n_chunk if type == "corr" else self.occ_n_chunk
        n_chunk_file = min(max_chunk, n_chunk_file)
        if random_flag:
            file_ind = np.random.permutation(max_chunk)[:n_chunk_file]
        else:
            file_ind = [i for i in range(n_chunk_file)]
        return file_ind

    def load(self, dir, id, n=None, type="occ", file_ind=None, random_flag=True):
        assert type in ["occ", "corr"]
        keys = ["uni_xyz", "nss_xyz", "uni_occ", "nss_occ"] if type == "occ" else ["arr_0"]
        if file_ind is None:
            file_ind = self.get_chunk_index(n, type, random_flag)
        data = {}
        for ind in file_ind:
            _data = np.load(join(dir, f"{id}_{ind}.npz"))
            for k in keys:
                if k not in data.keys():
                    data[k] = [_data[k]]
                else:
                    data[k].append(_data[k])
        for k in keys:
            data[k] = np.concatenate(data[k], axis=0)
        return data

    def __getitem__(self, index: int):
        ret = {}
        meta_info = self.meta_list[index]
        start = meta_info["start"]
        if self.input_type == "pcl":
            viz_id = f"{self.mode}_{os.path.basename(meta_info['seq_dir'])}_start{start}_{index}"
        else:
            viz_id = f"{self.mode}_{os.path.basename(meta_info['seq_dir'])}_{meta_info['view']}_start{start}_{index}"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode

        seq_dir = meta_info["seq_dir"]
        base_root = join(self.data_root, seq_dir)

        load_pcl_flag = self.input_type == "pcl"

        # load inputs
        inputs = []
        if load_pcl_flag:  # camera view observation
            object_T = np.eye(4)
        else:
            for d in range(self.seq_len):
                fn = join(base_root, meta_info["view"], f"{d+start}.npz")
                _data = np.load(fn)
                if d == 0:  # use the first frame as the camera frame
                    if self.use_camera_frame:
                        object_T = _data["object_T"]
                    else:
                        object_T = np.eye(4)
                canonical_view_pc = _data["canonical_view_pc"]
                choice = np.random.choice(canonical_view_pc.shape[0], self.n_inputs, replace=True)
                _in = canonical_view_pc[choice, :]
                _in = (object_T[:3, :3] @ _in.T + object_T[:3, 3:4]).T
                inputs.append(_in[None, ...])
        ret["object_T"] = object_T

        # load corr pc (and inputs)
        pointcloud = []
        points_mesh = []
        choice_corr, choice_inputs = None, None
        if self.mode == "train":
            if load_pcl_flag:
                pc_n = self.n_corr + self.n_inputs
            else:
                pc_n = self.n_corr
        else:
            if load_pcl_flag:
                pc_n = max(self.n_query_eval, self.n_corr + self.n_inputs)
            else:
                pc_n = max(self.n_query_eval, self.n_corr)
        corr_file_ind = self.get_chunk_index(pc_n, "corr", self.mode == "train")
        for i in range(self.seq_len):
            pc = self.load(join(base_root, "corr"), i + start, type="corr", file_ind=corr_file_ind)[
                "arr_0"
            ]
            pc = (object_T[:3, :3] @ pc.T + object_T[:3, 3:4]).T
            # append corr pts
            if choice_corr is None:
                if self.mode == "train":
                    choice_corr = np.random.choice(pc.shape[0], self.n_corr, replace=False)
                else:
                    choice_corr = np.array([i for i in range(self.n_corr)])
            pointcloud.append(pc[choice_corr][np.newaxis, ...])
            # if set, append inputs pts
            if load_pcl_flag:
                if choice_inputs is None:
                    if self.mode == "train":
                        choice_inputs = np.random.choice(pc.shape[0], self.n_inputs, replace=False)
                    else:
                        choice_inputs = np.array([i + self.n_corr for i in range(self.n_inputs)])
                inputs.append(pc[choice_inputs][np.newaxis, ...])
            # append mesh_points
            if self.mode != "train":
                points_mesh.append(pc[np.newaxis, : self.n_query_eval, :])

        inputs = np.concatenate(inputs, axis=0)
        noise = self.inputs_noise_std * np.random.randn(*inputs.shape)
        noise = noise.astype(np.float32)
        inputs = inputs + noise
        ret["inputs"] = inputs
        time = np.linspace(start=0.0, stop=1.0, num=self.seq_len)
        ret["inputs.time"] = time
        pointcloud = np.concatenate(pointcloud, axis=0)
        ret["pointcloud"] = pointcloud
        ret["pointcloud.time"] = time
        if len(points_mesh) > 0:
            points_mesh = np.concatenate(points_mesh)
            ret["points_mesh"] = points_mesh

        # load IF
        # determine which frame to sample
        if self.mode == "train":
            if self.oflow_flag:
                ind_list = np.array(
                    [0, int(np.random.choice(self.seq_len - 1, 1, replace=False) + 1)]
                )
            else:
                ind_list = np.random.choice(self.seq_len, self.n_training_frames, replace=False)
                ind_list.sort()
        else:
            ind_list = np.array([i for i in range(self.seq_len)])
        queries, occ_state = [], []
        for i in ind_list:
            if self.mode == "train":
                _occ_data = self.load(
                    join(base_root, "c_occ"), i + start, max(self.n_nss, self.n_uni), type="occ"
                )
                # load nss and uniform
                ns = _occ_data["nss_xyz"]
                choice = np.random.choice(ns.shape[0], self.n_nss, replace=False)
                ns = ns[choice, :]
                ns_o = np.unpackbits(_occ_data["nss_occ"])[choice]
                un = _occ_data["uni_xyz"]
                choice = np.random.choice(un.shape[0], self.n_uni, replace=False)
                un = un[choice, :]
                un_o = np.unpackbits(_occ_data["uni_occ"])[choice]
                _q = np.concatenate([un, ns], axis=0)
                _o = np.concatenate([un_o, ns_o], axis=0)
            else:
                # only load uniforms
                _occ_data = self.load(
                    join(base_root, "c_occ"),
                    i + start,
                    self.n_query_eval,
                    type="occ",
                    random_flag=False,
                )
                _q = _occ_data["uni_xyz"][: self.n_query_eval]
                _o = np.unpackbits(_occ_data["uni_occ"][: self.n_query_eval])
            # transform the canonical view occ queries
            _q = (object_T[:3, :3] @ _q.T + object_T[:3, 3:4]).T
            queries.append(_q[np.newaxis, ...])
            occ_state.append(_o[np.newaxis, ...])
        points = np.concatenate(queries, axis=0)
        occ = np.concatenate(occ_state, axis=0).astype(float)
        if self.mode == "train" and self.oflow_flag:
            ret["points"] = points[0]
            ret["points.occ"] = occ[0]
            ret["points.time"] = time[ind_list][0]
        else:
            ret["points"] = points
            ret["points.occ"] = occ
            ret["points.time"] = time[ind_list]
        ret["points_t"] = points[-1]
        ret["points_t.occ"] = occ[-1]
        ret["points_t.time"] = time[ind_list][-1]

        return ret, meta_info
