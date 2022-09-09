from .youtube_vis import YouTubeVIS
import os
from .. import utils
import numpy as np


class OccludedVIS(YouTubeVIS):

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/occluded_vis/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/occluded_vis/'),
            # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
            'SPLIT_TO_EVAL': 'train_maxlen200_first_frame',  # Valid: 'train_maxlen200_first_frame'
            'PRINT_CONFIG': True,  # Whether to print current config
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    def __init__(self, config=None, gt=None, predictions=None):
        super().__init__(config=config, tracker_name_file="occluded_vis_", gt_name_file="occluded_vis_", gt=gt, predictions=predictions)

    def _prepare_gt_annotations(self):
        """
        Prepares GT data by rle encoding segmentations and computing the average track area.
        :return: None
        """
        # only loaded when needed to reduce minimum requirements

        for track in self.gt_data['annotations']:
            areas = [a for a in track['areas'] if a]
            if len(areas) == 0:
                track['area'] = 0
            else:
                track['area'] = np.array(areas).mean()
