# ------------------------------------------------------------------------------
# Written by Zheng Tang (tangzhengthomas@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .veri import VeRiDataset as veri
from .cityflow import CityFlowDataset as cityflow
from .synthetic import SyntheticDataset as synthetic
from .veri_synthetic import VeRiSyntheticDataset as veri_synthetic
from .cityflow_synthetic import CityFlowSyntheticDataset as cityflow_synthetic