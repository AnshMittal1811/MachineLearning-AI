from logger.logger_meta.video_logger import VideoLogger
from .metric_logger import MetricLogger
from .image_logger import ImageLogger
from .checkpoint_logger import CheckpointLogger
from .xls_logger import XLSLogger
from .mesh_logger import MeshLogger
from .hist_logger import HistLogger

LOGGER_REGISTED = {
    "metric": MetricLogger,
    "image": ImageLogger,
    "checkpoint": CheckpointLogger,
    "xls": XLSLogger,
    "mesh": MeshLogger,
    "hist": HistLogger,
    "video": VideoLogger
}
