# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import shutil
from sys import argv

import xml.etree.ElementTree as ET

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import os
import os.path as osp
import numpy as np
import scipy.io as sio
import tqdm
from detectron2.config import get_cfg
from detectron2.structures import Boxes
import torch
from detectron2.projects import point_rend
import pycocotools.coco as coco

from .box2mask import Box2Mask

data_dir = '../data/100doh/'
anno_file = '/glusterfs/yufeiy2/download_data/COCO/annotations/instances_val2017.json'
coco_data = coco.COCO(anno_file)
coco_classes = coco_data.loadCats(coco_data.getCatIds())
coco_classes = [cat['name'] for cat in coco_classes]
raw_dir = '/home/yufeiy2/hoi_vid/output/100doh_clips'
odir = '../output/100doh_detectron/by_obj/'

def setup_model():
    detbase='../lasr/detectron2/' # sys.argv[2]
    cfg = get_cfg()

    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.
    cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

    predictor = Box2Mask(cfg)
    return predictor


def test(seqname, predictor):
    # detect the first frame, move object seq to odir 
    # rename seq to %05d from 0
    imgdir= '%s/JPEGImages/%s'%(odir,seqname)
    maskdir='%s/Annotations/%s'%(odir,seqname)

    root = load_anno(seqname)
    if root is None:
        return

    gt_boxes, is_object = find_gt_boxes(root)
    path = osp.join(data_dir, 'JPEGImages', seqname + '.jpg')
    img = cv2.imread(path)

    predictions = predictor(img, gt_boxes, is_object)

    segs = predictions['instances'].to('cpu')
    masks = segs.pred_masks.cpu().detach().numpy()
    gt_boxes = gt_boxes.tensor.cpu().numpy()
    is_object = is_object.numpy()
    hand_inds = np.where(is_object == 0)[0]
    obj_inds = np.where(is_object == 1)[0]
    hand_mask = ((masks[hand_inds] > 0) * 255).astype(np.uint8)
    obj_mask = ((masks[obj_inds] > 0) * 255).astype(np.uint8)

    # v = Visualizer(img, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    # vis = v.draw_instance_predictions(segs)
    # print('write to pred', out_pref)
    # cv2.imwrite(out_pref + '_pred.png', vis.get_image())

    for o in range(len(obj_mask)):
        os.makedirs(maskdir + '_%d' % o, exist_ok=True)
        cv2.imwrite(maskdir + '_%d/%05d.png' % (o, 0), obj_mask[o])
        # cv2.imwrite(maskdir + '_%d/%05d.png' % (o, 0), hand_mask[o])
        sio.savemat(maskdir + '_%d/%05d.mat' % (o, 0), 
            {'hand_mask': hand_mask, 'hand_box': gt_boxes[hand_inds],
            'obj_box': gt_boxes[obj_inds[o]]})

        dst_folder = imgdir + '_%d' % o
        if osp.exists(dst_folder): shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        src_folder = osp.join(raw_dir, '%s/frames/*.jp*' % seqname)
        for i,src_path in enumerate(sorted(glob.glob(src_folder))):
            shutil.copyfile(src_path, osp.join(dst_folder, '%05d.jpg' % i))


def find_gt_boxes(root, filter_field=None):  
    """
    filter_field: targetobject / hand
    """
    gt_boxes, is_object = [], []
    for obj in root.findall('object'):
        cls = obj.find('name')
        if filter_field is None or cls.text == filter_field:
            gt_bbox = extract_bbox(obj)
            gt_boxes.append(gt_bbox)
            is_object.append(cls.text == 'targetobject')
    gt_boxes = Boxes(torch.FloatTensor(gt_boxes))
    is_object = torch.LongTensor(is_object)
    return gt_boxes, is_object


def extract_bbox(obj):
    box = obj.find('bndbox')
    x1 = float(box.find('xmin').text)
    x2 = float(box.find('xmax').text)
    y1 = float(box.find('ymin').text)
    y2 = float(box.find('ymax').text)
    return [x1, y1, x2 ,y2]


def load_anno(index):
    """Images"""
    path = os.path.join(data_dir, 'Annotations', index + '.xml')
    if not osp.exists(path):
        print (path)
        return None
    with open(path, 'r') as fid:
        root = ET.parse(fid).getroot()
    return root


if __name__ == '__main__':
    predictor = setup_model()
    seqname = argv[1] if len(argv) >= 2 else '*'
    vid_list = glob.glob(osp.join(raw_dir, seqname))
    vid_list = [osp.basename(e) for e in vid_list]
    print('Running detectron2 to extract first frame')
    print('save to ', odir)
    for vid in tqdm.tqdm(vid_list):
        test(vid, predictor)
    print('!save to ', odir)
    