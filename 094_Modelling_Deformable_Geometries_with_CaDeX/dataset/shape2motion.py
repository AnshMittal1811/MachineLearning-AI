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

        self.n_chunk_dict = {
            "occ": cfg["dataset"]["occ_n_chunk"],
            "corr": cfg["dataset"]["corr_n_chunk"],
        }
        if "pc_n_chunk" in cfg["dataset"].keys():
            self.n_chunk_dict["pc"] = cfg["dataset"]["pc_n_chunk"]
        else:
            self.n_chunk_dict["pc"] = self.n_chunk_dict["corr"]
        self.chunk_size = cfg["dataset"]["chunk_size"]

        self.set_size = cfg["dataset"]["set_size"]

        # self.pcl_traj = cfg["dataset"]["pcl_traj"]
        self.use_camera_frame = cfg["dataset"]["camera_frame"]

        self.num_theta = cfg["dataset"]["num_atc"]
        self.n_inputs = cfg["dataset"]["num_input_pts"]
        self.inputs_noise_std = cfg["dataset"]["input_noise"]
        self.n_uni = cfg["dataset"]["num_query_uni"]
        self.n_nss = cfg["dataset"]["num_query_ns"]
        self.n_corr = cfg["dataset"]["num_corr_pts"]
        self.n_query_eval = cfg["dataset"]["n_query_sample_eval"]
        if self.mode != "train":
            logging.info(f"{self.mode} eval_n set to {self.n_query_eval}")
            logging.info("Be careful! Test eval_n should set to 100000")

        # build meta info
        self.one_per_instance = False
        if "one_per_instance" in cfg["dataset"].keys():
            self.one_per_instance = cfg["dataset"]["one_per_instance"]
        self.meta_list = []
        self.input_type = cfg["dataset"]["input_type"]
        assert self.input_type in ["dep", "pcl"]
        split_fn = cfg["dataset"]["split"][self.mode]
        with open(join(self.data_root, split_fn)) as f:
            split_data = json.load(f)

        for dp in split_data:
            # * filter DP
            if "filter" in cfg["dataset"].keys():
                theta_range = cfg["dataset"]["filter"]
                _filtered_dp = []
                for _d in dp:
                    if self.num_theta == 1:
                        theta = [float(_d.split(".")[0].split("art")[-1])]
                    else:
                        theta = [
                            float(_d.split(".")[0].split("art")[-1][:4]),
                            float(_d.split(".")[0].split("art")[-1][4:]),
                        ]
                    for _i in range(self.num_theta):
                        if theta[_i] >= theta_range[_i][0] and theta[_i] <= theta_range[_i][1]:
                            _filtered_dp.append(_d)
                dp = _filtered_dp

            obj_id = dp[0].split("art")[0] + "art"
            sub_dir = os.path.join(self.data_root, obj_id)
            if not os.path.exists(sub_dir):
                logging.warning(f"{sub_dir} not exists, skip!")
                raise RuntimeError("Dataset broken!")
                continue

            with open(os.path.join(sub_dir, "meta.json")) as f:
                meta_data = json.load(f)
            num_views = meta_data["num_views"]
            T = len(dp)
            for t in range(T):
                if t + self.set_size <= T:
                    meta = {
                        "dir": sub_dir,
                        "files": [it.split(".")[0] for it in dp[t : t + self.set_size]],
                    }
                    if self.input_type == "pcl":  # spare point-cloud input
                        self.meta_list.append(meta)
                    else:
                        for view_id in range(num_views):
                            self.meta_list.append({**meta, "view": view_id})
                else:
                    break
                if self.one_per_instance:
                    break

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
        assert type in self.n_chunk_dict.keys()
        n_chunk_file = int(np.ceil(n / float(self.chunk_size)))
        max_chunk = self.n_chunk_dict[type]
        n_chunk_file = min(max_chunk, n_chunk_file)
        if random_flag:
            file_ind = np.random.permutation(max_chunk)[:n_chunk_file]
        else:
            file_ind = [i for i in range(n_chunk_file)]
        return file_ind

    def load(self, dir, id, n=None, type="occ", file_ind=None, random_flag=True):
        assert type in ["occ", "corr", "pc"]
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
        if self.input_type == "pcl":
            viz_id = f"{self.mode}_{os.path.basename(meta_info['dir'])}_idx{index}"
        else:
            viz_id = (
                f"{self.mode}_{os.path.basename(meta_info['dir'])}_{meta_info['view']}_idx{index}"
            )
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode

        base_root = meta_info["dir"]
        load_pcl_flag = self.input_type == "pcl"

        # load inputs
        inputs = []
        if load_pcl_flag:  # camera view observation
            object_T = np.eye(4)
        else:
            for d in range(self.set_size):
                fn = join(base_root, "obs", f"{meta_info['files'][d]}_{meta_info['view']}.npz")
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

        # load PC and points mesh
        points_chamfer = []
        if self.mode == "train":
            pc_all_n = self.n_inputs
        else:
            pc_all_n = max(self.n_inputs, self.n_query_eval)
        for i in range(self.set_size):
            pc = self.load(
                join(base_root, "pc"),
                meta_info["files"][i],
                n=pc_all_n,
                type="pc",
                random_flag=self.mode == "train",
            )["arr_0"]
            pc = (object_T[:3, :3] @ pc.T + object_T[:3, 3:4]).T
            # if set, append inputs pts
            if load_pcl_flag:
                if self.mode == "train":
                    choice_inputs = np.random.choice(pc.shape[0], self.n_inputs, replace=False)
                else:
                    choice_inputs = np.array([i for i in range(self.n_inputs)])
                inputs.append(pc[choice_inputs][np.newaxis, ...])
            # append mesh_points
            if self.mode != "train":
                points_chamfer.append(pc[np.newaxis, : self.n_query_eval, :])

        inputs = np.concatenate(inputs, axis=0)
        noise = self.inputs_noise_std * np.random.randn(*inputs.shape)
        noise = noise.astype(np.float32)
        inputs = inputs + noise
        ret["inputs"] = inputs
        if len(points_chamfer) > 0:
            points_chamfer = np.concatenate(points_chamfer)
            ret["points_chamfer"] = points_chamfer  # ! this is for CD eval

        # load corr pc
        pointcloud = []
        points_mesh = []
        choice_corr = None
        if self.mode == "train":
            corr_all_n = self.n_corr
        else:
            corr_all_n = max(self.n_query_eval, self.n_corr)
        corr_file_ind = self.get_chunk_index(corr_all_n, "corr", self.mode == "train")
        for i in range(self.set_size):
            pc = self.load(
                join(base_root, "corr"), meta_info["files"][i], type="corr", file_ind=corr_file_ind
            )["arr_0"]
            pc = (object_T[:3, :3] @ pc.T + object_T[:3, 3:4]).T
            # append corr pts
            if choice_corr is None:
                if self.mode == "train":
                    choice_corr = np.random.choice(pc.shape[0], self.n_corr, replace=False)
                else:
                    choice_corr = np.array([i for i in range(self.n_corr)])
            pointcloud.append(pc[choice_corr][np.newaxis, ...])
            if self.mode != "train":
                points_mesh.append(pc[np.newaxis, : self.n_query_eval, :])
        pointcloud = np.concatenate(pointcloud, axis=0)
        ret["pointcloud"] = pointcloud
        if len(points_mesh) > 0:
            points_mesh = np.concatenate(points_mesh)
            ret["points_mesh"] = points_mesh  # ! this is for CORR eval

        # load IF
        queries, occ_state = [], []
        for i in range(self.set_size):
            if self.mode == "train":
                _occ_data = self.load(
                    join(base_root, "implicit"),
                    meta_info["files"][i],
                    max(self.n_nss, self.n_uni),
                    type="occ",
                )
                # load nss and uniform
                ns = _occ_data["nss_xyz"]
                choice = np.random.choice(ns.shape[0], self.n_nss, replace=False)
                ns = ns[choice, :]
                ns_o = (_occ_data["nss_occ"] < 0).astype(float)[choice]
                un = _occ_data["uni_xyz"]
                choice = np.random.choice(un.shape[0], self.n_uni, replace=False)
                un = un[choice, :]
                un_o = (_occ_data["uni_occ"] < 0).astype(float)[choice]
                assert (un_o.sum() / un_o.shape[0]) < (
                    ns_o.sum() / ns_o.shape[0]
                ), "NS rate < UNI rate, This happens very rare, please check the dataset! Process Stopped"
                _q = np.concatenate([un, ns], axis=0)
                _o = np.concatenate([un_o, ns_o], axis=0)
            else:
                # only load uniforms
                _occ_data = self.load(
                    join(base_root, "implicit"),
                    meta_info["files"][i],
                    self.n_query_eval,
                    type="occ",
                    random_flag=False,
                )
                _q = _occ_data["uni_xyz"][: self.n_query_eval]
                _o = (_occ_data["uni_occ"] < 0).astype(float)[: self.n_query_eval]
            # transform the canonical view occ queries
            _q = (object_T[:3, :3] @ _q.T + object_T[:3, 3:4]).T
            queries.append(_q[np.newaxis, ...])
            occ_state.append(_o[np.newaxis, ...])
        points = np.concatenate(queries, axis=0)
        occ = np.concatenate(occ_state, axis=0).astype(float)

        ret["points"] = points
        ret["points.occ"] = occ
        # ret["points_t"] = points[-1]
        # ret["points_t.occ"] = occ[-1]

        assert len(meta_info["files"][0].split("art")[-1]) == 4 * self.num_theta
        if self.num_theta == 1:
            theta = [[float(f.split("art")[-1])] for f in meta_info["files"]]
        else:
            theta = [
                [float(f.split("art")[-1][:4]), float(f.split("art")[-1][4:])]
                for f in meta_info["files"]
            ]
        ret["theta"] = np.array(theta) / 180.0 * np.pi

        # fake time for LPDC
        ret["fake_time"] = np.linspace(start=0.0, stop=1.0, num=self.set_size)
        return ret, meta_info
