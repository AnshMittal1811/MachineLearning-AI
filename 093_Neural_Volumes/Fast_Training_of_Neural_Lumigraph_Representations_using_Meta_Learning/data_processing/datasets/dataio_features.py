from pathlib import Path

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio
from pypfm import PFMLoader


class DatasetFlyingChairs2(Dataset):
    """
    Flying Chairs 2 Dataset
    """

    def __init__(self, dataset_path: Path, opt):
        super().__init__()

        self.opt = opt

        self.dataset_root_path = dataset_path
        filenames = os.listdir(dataset_path)

        self.dataset_item_paths = {
            'img0_paths': sorted(filter(lambda x: x[-9:-4] == "img_0", filenames)),
            'img1_paths': sorted(filter(lambda x: x[-9:-4] == "img_1", filenames)),
            'flow01_paths': sorted(filter(lambda x: x[-11:-4] == "flow_01", filenames)),
            'flow10_paths': sorted(filter(lambda x: x[-11:-4] == "flow_10", filenames)),
            'occ01_paths': sorted(filter(lambda x: x[-10:-4] == "occ_01", filenames)),
            'occ10_paths': sorted(filter(lambda x: x[-10:-4] == "occ_10", filenames)),
            'occw01_paths': sorted(filter(lambda x: x[-18:-4] == "occ_weights_01", filenames)),
            'occw10_paths': sorted(filter(lambda x: x[-18:-4] == "occ_weights_10", filenames))
        }

    def __len__(self):
        return len(self.dataset_item_paths['img0_paths'])

    def __getitem__(self, idx):
        item_paths = self.get_item_filepaths(idx)
        item_dict = self.get_items_from_filepaths(item_paths)

        item_dict_torch = {}
        for key, value in item_dict.items():
            value_torch = torch.from_numpy(value).float()
            if len(value_torch.shape) < 3:
                item_dict_torch[key] = value_torch.unsqueeze(0)
            else:
                item_dict_torch[key] = value_torch.permute(2, 0, 1)

        return item_dict_torch

    def get_item_filepaths(self, idx):
        paths = {}
        for key, value in self.dataset_item_paths.items():
            paths[key.replace('_paths', '')] = os.path.join(self.dataset_root_path, value[idx])
        return paths

    def get_items_from_filepaths(self, item_paths):
        item_dict = {}
        for key, value in item_paths.items():
            if key in ['flow01', 'flow10']:
                flow_pixels = np.asarray(self.read_flow(value))
                # flow_normalized = np.stack([2 * flow_pixels[..., 0] / flow_pixels.shape[1],
                #                             2 * flow_pixels[..., 1] / flow_pixels.shape[0]], axis=-1)
                # item_dict[key] = flow_normalized
                item_dict[key] = flow_pixels
            elif key in ['img0', 'img1', 'occ01', 'occ10']:
                item_dict[key] = np.asarray(imageio.imread(value)) / 255.
            elif key in ['occw01', 'occw10']:
                item_dict[key] = np.asarray(self.load_pfm(value))
        return item_dict

    def read_flow(self, file):
        assert type(file) is str, "file is not str %r" % str(file)
        assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
        assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
        f = open(file, 'rb')
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % flo_number
        w, h = np.fromfile(f, np.int32, count=2)
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = np.resize(data, (int(h), int(w), 2))
        f.close()

        return flow

    def load_pfm(self, file):
        loader = PFMLoader(color=False, compress=False)
        return loader.load_pfm(file)

