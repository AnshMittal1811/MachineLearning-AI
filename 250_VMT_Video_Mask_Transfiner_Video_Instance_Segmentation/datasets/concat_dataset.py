from pathlib import Path

import torch
import torch.utils.data

from util.misc import get_local_rank, get_local_size
import datasets.transforms_clip as T
from torch.utils.data import Dataset, ConcatDataset
from .coco2seq import build as build_seq_coco
from .ytvos import build as build_ytvs



def build(image_set, args):
    print('preparing coco2seq dataset ....')
    coco_seq =  build_seq_coco(image_set, args)
    print('preparing hq ytvis dataset  .... ')
    ytvis_dataset = build_ytvs(image_set, args)

    concat_data = ConcatDataset([ytvis_dataset, coco_seq])

    return concat_data


