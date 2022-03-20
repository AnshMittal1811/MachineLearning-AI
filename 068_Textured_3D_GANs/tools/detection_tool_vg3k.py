from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import numpy as np

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='configs/bbox2mask_vg/eval_sw_R101/runtest_clsbox_2_layer_mlp_nograd_R101.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='lib/datasets/data/trained_models/33219850_model_final_coco2vg3k_seg.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        required=True,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: *)',
        default='*',
        type=str
    )
    parser.add_argument(
        'input_folder', help='folder of images', default=None
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--thresh',
        default=0.0,
        type=float,
        help='filter predictions using threshold (set to 0 to save everything)',
    )
    parser.add_argument(
        '--filter-classes',
        type=str,
        help='filter classes using hardcoded class list (comma-separated indices)',
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def convert(cls_boxes, cls_segms):
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, classes

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = (
        dummy_datasets.get_vg3k_dataset()
        if args.use_vg3k else dummy_datasets.get_coco_dataset())

    assert os.path.isdir(args.input_folder), 'The input must be a folder!'
    im_list = glob.glob(args.input_folder + '/*.' + args.image_ext)
    im_list += glob.glob(args.input_folder + '/*/*.' + args.image_ext) # Add subdirectories
    im_list = sorted(im_list)
    print('Got', len(im_list), 'files to process!')

    if args.filter_classes is not None:
        filter_classes = [int(x) for x in args.filter_classes.split(',')]
        print('Filtering classes', filter_classes)
    else:
        filter_classes = None
        print('Not filtering classes.')
        
    print('Saving to', args.output_dir)

    for i, im_name in enumerate(im_list):
        logger.info('Processing {}'.format(im_name))
        im = cv2.imread(im_name)
        if im is None:
            logger.info('Unable to read image, skipping.')
            continue
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        boxes, segms, classes = convert(cls_boxes, cls_segms)
        classes = np.array(classes, dtype=np.uint16)
        resolution = segms[0]['size']
        segms = np.array([x['counts'] for x in segms]) # Run-length encoding
        
        valid = boxes[:, 4] >= args.thresh
        if filter_classes is not None:
            valid &= np.isin(classes, filter_classes)
            
        boxes = boxes[valid].copy()
        classes = classes[valid].copy()
        segms = segms[valid].copy()
        
        rel_path = im_name.replace(args.input_folder + '/', '').replace(args.input_folder + '\\', '')
        output_full_path = os.path.join(args.output_dir, rel_path)
        basedir = os.path.dirname(output_full_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        
        np.savez_compressed(output_full_path, boxes=boxes, segments=segms, classes=classes, resolution=resolution)


if __name__ == '__main__':
    if sys.version_info.major != 2:
        print('This script must be run using Python 2!')
        sys.exit(1)
    
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
