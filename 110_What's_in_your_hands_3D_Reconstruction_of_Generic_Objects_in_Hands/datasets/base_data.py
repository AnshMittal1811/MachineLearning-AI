# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os.path as osp
import torch
from torch.utils.data import Dataset

class BaseData(Dataset):
    def __init__(self, cfg, dataset: str, split='val', is_train=True,
                 data_dir='../data/'):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.is_train = is_train
        self.data_dir = data_dir

        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],
            'hTo': [],
        }
        self.obj2mesh = {}
        self.map = None
        
    def preload_anno(self, load_keys=['index']):
        raise NotImplementedError

    def get_sdf_files(self, cad_idx):
        sdf_dir = osp.join(self.cfg.DB.DIR, 'sdf/SdfSamples/', self.dataset, 'all')
        filename = osp.join(sdf_dir, cad_idx + '.npz')
        assert osp.exists(filename), 'Not exists %s' % filename
        return filename

    def __len__(self, ):
        return len(self.anno['index'])


def minmax(pts2d):
    x1y1 = torch.min(pts2d, dim=-2)[0]  # (..., P, 2)
    x2y2 = torch.max(pts2d, dim=-2)[0]  # (..., P, 2)
    return torch.cat([x1y1, x2y2], dim=-1)  # (..., 4)


def proj3d(points, cam):
    p2d = points.cpu() @ cam.cpu().transpose(-1, -2)
    p2d = p2d[..., :2] / p2d[..., 2:3]
    return p2d    