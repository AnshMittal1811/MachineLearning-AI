from .sort import Sort
import os
import numpy as np
import pickle
import slowfast.utils.logging as logging
import slowfast.utils.distributed as du

AVA_VALID_SECS = range(902, 1799)

logger = logging.get_logger(__name__)


# split to vid and frame
FPS=30
START_SEC = 900
END_SEC = 1799
AVA_VALID_SECS = range(START_SEC, END_SEC)
AVA_FRAMES_RANGE = range(FPS, (END_SEC - START_SEC) * FPS, FPS)
BASE_DIR = os.path.join('run_files', 'linked_boxes')
def sec_to_frame(sec):
    """
    Convert time index (in second) to frame index.
    0: 900
    30: 901
    """
    sec = int(sec)
    return (sec - 900) * FPS

def gen_boxes_dict(all_pbboxes, is_gt):
    dboxes = {}
    if isinstance(all_pbboxes, str):
        all_pbboxes = [all_pbboxes]
    for pbboxes in all_pbboxes:
        with open(pbboxes, 'rt') as f: bboxes = f.read().split('\n')

        for l in bboxes[:-1]:
            l = l.split(',')
            if is_gt:
                vid, sec, x1,y1,x2,y2,label, box_id = l
                score = 1
            else:
                vid, sec, x1,y1,x2,y2,label, score = l
            sec = (int(sec) - START_SEC) * FPS
            if vid not in dboxes: dboxes[vid] = {}
            if sec not in dboxes[vid]: dboxes[vid][sec] = []
            dboxes[vid][sec].append(list(map(float, [x1,y1,x2,y2,score])))
    return dboxes

def sort_boxes_dict(dboxes, clear_dups_threshold):
    ret = {}
    n = len(dboxes)
    for i, vid in enumerate(dboxes.keys()):
        if i%10 == 0: print(f">> {i}/{n}", end = ', ')
        ret[vid] = get_vid_boxes_linked(vid, dboxes, clear_dups_threshold)
    print()
    return ret

def get_vid_boxes_linked(vid, dboxes, clear_dups_threshold):
    osort = Sort(clear_dups_threshold=clear_dups_threshold)
    dvid = dboxes[vid]
    dvid_sorted = {}
    rng = AVA_FRAMES_RANGE
    for fid in rng:
        boxes = dvid.get(fid, np.empty((0, 5)))
        boxes = np.array(boxes)
        dvid_sorted[fid] = osort.update(boxes)
    return dvid_sorted

def get_out_path(pbboxes):
    if isinstance(pbboxes, str): pbboxes = [pbboxes]
    pbboxes = sorted(pbboxes)
    ret = ''
    for p in pbboxes:
        path = p.replace('/', '__')
        ret = ret + '_' + path
    os.makedirs(BASE_DIR, exist_ok=True, mode=0o777)
    ret = os.path.join(BASE_DIR, ret)
    return path

def get_ava_boxes(pbboxes):
    if isinstance(pbboxes, str): pbboxes = [pbboxes]
    out_path = get_out_path(pbboxes)
    if not os.path.exists(out_path) and du.is_master_proc():
        logger.info(f"Generating Linked Boxes file: {pbboxes}")
        is_gt = 'detect' not in pbboxes
        dboxes = gen_boxes_dict(pbboxes, is_gt)
        clear_dups_threshold = 0 if is_gt else 0.7
        sorted_boxes = sort_boxes_dict(dboxes, clear_dups_threshold)
        with open(out_path, 'wb') as f:
            pickle.dump(sorted_boxes, f)
    du.synchronize()
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            ret = pickle.load(f)
    return ret


if __name__ == '__main__':
    base = '/home/gamir/DER-Roei/datasets/AVA/download_repo/data/ava'
    # pbboxes = 'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv'
    # pbboxes = 'annotations/ava_train_v2.2.csv'
    pbboxes = "annotations/person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"
    pbboxes = os.path.join(base, pbboxes)
    out_path = get_out_path(pbboxes)
    is_gt = 'detect' not in pbboxes
    print(f"START GEN DBOXES, is_get: {is_gt}, pbboxes: {pbboxes}")
    dboxes = gen_boxes_dict(pbboxes, is_gt)
    clear_dups_threshold = 0 if is_gt else 0.7
    print(f"GENERATED DBOXES")
    sorted_boxes = sort_boxes_dict(dboxes, clear_dups_threshold)
    print(f"SAVE TO {out_path}")
    with open(out_path, 'wb') as f: pickle.dump(sorted_boxes, f)
    print(f"DONE")