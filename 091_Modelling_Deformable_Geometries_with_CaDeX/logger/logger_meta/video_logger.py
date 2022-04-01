from .base_logger import BaseLogger
import matplotlib
import os
import numpy as np
from cv2 import imwrite
import cv2
import torch
from copy import deepcopy
from PIL import Image
import imageio


class VideoLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "video"
        os.makedirs(self.log_path, exist_ok=True)
        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        return

    def log_batch(self, batch):

        # get data
        if not self.NAME in batch["output_parser"].keys():
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
        for video_key in keys_list:  # for each key
            if video_key not in data.keys():
                continue
            kdata = data[video_key]
            assert len(kdata.shape) == 5
            nbatch = kdata.shape[0]
            if isinstance(kdata, torch.Tensor):
                kdata = kdata.detach().cpu().numpy()
            # for each sample in batch
            for batch_id in range(nbatch):
                # now all cases are converted to list of image
                vi = kdata[batch_id]  # T,3,H,W / T,1,H,W
                assert vi.ndim == 4
                # first process image
                color_flag = False
                if vi.shape[1] == 1:
                    color_flag = True
                    cm = matplotlib.cm.get_cmap("magma")  # ("viridis")
                    vi = cm(vi.squeeze(0))[..., :3]
                    vi = vi.transpose(0, 3, 1, 2)
                    vi *= 255
                else:
                    vi *= 255.0 if vi.max() < 200 else 1
                vi = np.clip(vi, a_min=0, a_max=255)
                vi = vi.astype(np.uint8)
                self.tb.add_video(
                    video_key + "/" + phase,
                    torch.LongTensor(vi).unsqueeze(0) / 255.0
                    if color_flag
                    else torch.LongTensor(vi).unsqueeze(0) / 255.0,
                    current_epoch,
                )
                # save to file
                frames = [Image.fromarray(f.transpose(1, 2, 0)) for f in vi]
                filename = os.path.join(
                    self.log_path,
                    "epoch_%d" % current_epoch,
                    meta_info["viz_id"][batch_id] + video_key + ".gif",
                )
                imageio.mimsave(filename, frames, duration=0.03 * len(frames))

                if self.viz_one:
                    break

    def log_phase(self):
        pass
