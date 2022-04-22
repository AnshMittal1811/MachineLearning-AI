import argparse
import os
import os.path as osp
from demo import demo_utils

import numpy as np
from PIL import Image

import sys
from nnutils.hand_utils import ManopthWrapper
sys.path.append('externals/frankmocap/')
sys.path.append('externals/frankmocap/detectors/body_pose_estimator/')

from renderer.screen_free_visualizer import Visualizer

from nnutils.handmocap import get_handmocap_predictor, process_mocap_predictions, get_handmocap_detector
from nnutils.hoiapi import get_hoi_predictor, vis_hand_object
from nnutils import box2mask
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer as VisDet
from detectron2.structures.boxes import BoxMode


def get_args():
    parser = argparse.ArgumentParser(description="Optimize object meshes w.r.t. human.")
    parser.add_argument(
        "--filename", default="demo/test.jpg", help="Path to image."
    )
    parser.add_argument("--out", default="output", help="Dir to save output.")
    parser.add_argument("--view", default="ego_centric", help="Dir to save output.")

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='weights/mow'
    )
    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):

    visualizer = Visualizer('pytorch3d')
    image = Image.open(args.filename).convert("RGB")
    image = np.array(image)
    print(image.shape)
    
    # precit hand
    bbox_detector = get_handmocap_detector(args.view)
    detect_output = bbox_detector.detect_hand_bbox(image[..., ::-1].copy())
    body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
    res_img = visualizer.visualize(image, hand_bbox_list = hand_bbox_list)
    demo_utils.save_image(res_img, osp.join(args.out, 'hand_bbox.jpg'))
    
    hand_predictor = get_handmocap_predictor()
    mocap_predictions = hand_predictor.regress(
        image[..., ::-1], hand_bbox_list
    )
    # MOW model also takes in masks but currently we feed in all 1. You could specify masks yourself, 
    # or if you have bounding box for object masks, we can convert it to masks 
    
    # mask_predictor = box2mask.setup_model()
    # boxes = # object_bbox
    # boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    # predictions, object_mask = mask_predictor(image[..., ::-1], boxes, pad=0)
    object_mask = np.ones_like(image[..., 0]) * 255

    # predict hand-held object
    hand_wrapper = ManopthWrapper().to('cpu')
    data = process_mocap_predictions(
        mocap_predictions, image, hand_wrapper, mask=object_mask
    )

    hoi_predictor = get_hoi_predictor(args)
    output = hoi_predictor.forward_to_mesh(data)
    vis_hand_object(output, data, image, args.out + '/%s' % osp.basename(args.filename).split('.')[0])
    

if __name__ == "__main__":
    main(get_args())
