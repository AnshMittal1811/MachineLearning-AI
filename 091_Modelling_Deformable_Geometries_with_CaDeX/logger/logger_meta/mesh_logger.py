from .base_logger import BaseLogger
import torch
import os
import trimesh
import numpy as np


class MeshLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "mesh"
        os.makedirs(self.log_path, exist_ok=True)
        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        return

    def log_batch(self, batch):

        # get data
        if self.NAME not in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch["data"]
        phase = batch["phase"]
        current_epoch = batch["epoch"]
        meta_info = batch["meta_info"]

        # check whether need log
        if not batch["viz_flag"]:
            return

        os.makedirs(os.path.join(self.log_path, "epoch_%d" % current_epoch), exist_ok=True)

        # log
        for mesh_key in keys_list:  # for each key
            if mesh_key not in data.keys():
                continue
            kdata = data[mesh_key]
            if isinstance(kdata, list):
                assert isinstance(kdata[0], trimesh.Trimesh)
                for i, mesh in enumerate(kdata):
                    viz_id = meta_info["viz_id"][i]
                    save_fn = os.path.join(
                        self.log_path, "epoch_%d" % current_epoch, mesh_key + "_" + viz_id + ".ply"
                    )
                    mesh.export(save_fn)
                    config_dict = {
                        "camera": {"cls": "PerspectiveCamera", "fov": 75},
                        "lights": [
                            {
                                "cls": "AmbientLight",
                                "color": "#ffffff",
                                "intensity": 0.7,
                            },
                            {
                                "cls": "DirectionalLight",
                                "color": "#ffffff",
                                "intensity": 0.65,
                                "position": [0, 2, 0],
                            },
                        ],
                        "material": {"cls": "MeshStandardMaterial", "roughness": 1, "metalness": 0},
                    }
                    self.tb.add_mesh(
                        tag=mesh_key + "/" + phase,
                        vertices=torch.Tensor(np.array(mesh.vertices)).unsqueeze(0),
                        faces=torch.Tensor(np.array(mesh.faces)).unsqueeze(0),
                        global_step=batch["batch"],
                        config_dict=config_dict,
                    )
                    if self.viz_one:
                        break
            elif isinstance(kdata, torch.Tensor):
                point_size_config = {"cls": "PointsMaterial", "size": 10}
                assert len(kdata.shape) == 3
                if kdata.shape[2] == 3:
                    for pc in kdata:
                        self.tb.add_mesh(
                            tag=mesh_key + "/" + phase,
                            vertices=pc.unsqueeze(0),
                            global_step=batch["batch"],
                            # config_dict={"material": point_size_config},
                        )
                        if self.viz_one:
                            break
                elif kdata.shape[2] == 6:
                    for pc in kdata:
                        self.tb.add_mesh(
                            tag=mesh_key + "/" + phase,
                            vertices=pc.unsqueeze(0)[..., :3],
                            colors=pc.unsqueeze(0)[..., 3:],
                            # config_dict={"material": point_size_config},
                            global_step=batch["batch"],
                        )
                        if self.viz_one:
                            break
                else:
                    raise RuntimeError("Point cloud logger accepts shape B,N,3/6")

    def log_phase(self):
        pass
