import numpy as np
import os
import shutil
import glob
from tqdm import tqdm
import scipy
import scipy.ndimage
import torch
import scipy.io as sio
import copy
import argparse
import pathlib

import detectron2
from detectron2 import model_zoo
from detectron2.projects import point_rend
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2 import structures
import cv2
import pycocotools.mask as mask_util

from data.vg3k_classes import vg3k_classes
from data.definitions import class_indices, dataset_to_class_name, vg3k_class_set, imagenet_synsets


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='all', help='all (default), imagenet, cub, p3d, or comma-separated')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use; multi-gpu not supported')
parser.add_argument('--detection_threshold', type=float, default=0.9, help='detector score threshold')
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

def load_cmr_paths(anno_path):
    anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
    gt = set()
    for im in anno:
        p = im.rel_path.replace('\\', '/')
        gt.add(p)
    return gt

class ImageList:
    def __iter__(self):
        raise NotImplementedError()
    
    def __len__(self):
        raise NotImplementedError()
        
    def __str__(self):
        raise NotImplementedError()
        

class ImageNetList(ImageList):
    def __init__(self, synsets):
        self.synsets = synsets

        self.paths = []
        self.detection_paths = []
        for synset in synsets:
            paths = sorted(glob.glob(f'datasets/imagenet/images/{synset}/*'))
            assert len(paths) > 0, f'Expected non-empty image directory for synset {synset}!'
            detection_paths = sorted(glob.glob(f'datasets/imagenet/vg3k_detections/{synset}/*.npz'))
            assert len(detection_paths) > 0, f'Expected non-empty directory for vg3k detections (synset {synset})!'
            assert len(detection_paths) == len(paths), f'Expected {len(paths)} detections, got {len(detection_paths)} (synset {synset})'
            
            # Check that images are aligned
            for p, dp in zip(paths, detection_paths):
                assert os.path.basename(p).split('.')[0] == os.path.basename(dp).split('.')[0], f'Filenames do not match! ({p}, {dp})'

            self.paths += paths
            self.detection_paths += detection_paths
        
        assert len(self.paths) == len(self.detection_paths)
    
    def __len__(self):
        return len(self.paths)
    
    def __iter__(self):
        return zip(iter(self.paths), iter(self.detection_paths))
    
        
class PascalImageNetList(ImageList):
    def __init__(self, class_name, cmr_subset=False):
        self.class_name = class_name
        
        self.root_dir = 'datasets/p3d/PASCAL3D+_release1.1/Images'
        self.detection_dir = 'datasets/p3d/vg3k_detections'
        self.image_dir = os.path.join(self.root_dir, f'{class_name}_imagenet')
        self.paths = sorted(glob.glob(os.path.join(self.image_dir, '*')))
        paths_set = set([os.path.basename(x) for x in self.paths])
        self.detection_paths = []
        assert len(self.paths) > 0
        
        if cmr_subset:
            cmr_class_name = class_name
            if cmr_class_name == 'airplane':
                cmr_class_name = 'aeroplane' # Rename
            cmr_paths = load_cmr_paths(f'datasets/p3d/data/{cmr_class_name}_train.mat')
            self.paths = []
            for path in cmr_paths:
                if '_imagenet' in path:
                    assert os.path.basename(path) in paths_set, path
                    self.paths.append(os.path.join(self.root_dir, path))
                    self.detection_paths.append(os.path.join(self.detection_dir, path + '.npz'))
                    # Check existence
                    assert os.path.isfile(self.detection_paths[-1]), self.detection_paths[-1]
                    
        
    def __len__(self):
        return len(self.paths)
        
    def __iter__(self):
        return zip(iter(self.paths), iter(self.detection_paths))
        
class CubImageList(ImageList):
    def __init__(self, train_only=True, cmr_subset=False):
        
        self.root_dir = 'datasets/cub/CUB_200_2011'
        self.detection_dir = 'datasets/cub/vg3k_detections'
        
        if cmr_subset:
            assert train_only
            cmr_paths = load_cmr_paths('datasets/cub/data/train_cub_cleaned.mat')
            self.paths = []
            self.detection_paths = []
            for path in cmr_paths:
                self.paths.append(os.path.join(self.root_dir, 'images', path))
                self.detection_paths.append(os.path.join(self.detection_dir, path + '.npz'))
                # Check existence
                assert os.path.isfile(self.paths[-1]), self.paths[-1]
                assert os.path.isfile(self.detection_paths[-1]), self.detection_paths[-1]
            
        else:
            is_train = set()
            with open(os.path.join(self.root_dir, 'train_test_split.txt'), 'r') as f:
                for line in f.readlines():
                    idx, split = line.strip().split(' ')
                    if split == '1':
                        is_train.add(idx)

            self.paths = []
            self.detection_paths = []
            with open(os.path.join(self.root_dir, 'images.txt'), 'r') as f:
                for line in f.readlines():
                    idx, path = line.strip().split(' ')
                    if idx in is_train or not train_only:
                        self.paths.append(os.path.join(self.root_dir, 'images', path))
                        self.detection_paths.append(os.path.join(self.detection_dir, path + '.npz'))
                        # Check existence
                        assert os.path.isfile(self.paths[-1]), self.paths[-1]
                        assert os.path.isfile(self.detection_paths[-1]), self.detection_paths[-1]
            
    def __len__(self):
        return len(self.paths)    
    
    def __iter__(self):
        return zip(iter(self.paths), iter(self.detection_paths))
    
class CustomImageList(ImageList):
    def __init__(self, dataset_name):
        print('Initializing custom dataset', dataset_name)
        
        self.paths = []
        self.detection_paths = []
        
        paths = glob.glob(f'datasets/{dataset_name}/images/*.*')
        if paths[0].endswith('json'):
            del paths[0] # Hacky
        paths += glob.glob(f'datasets/{dataset_name}/images/*/*.*') # Also search subdirectories
        paths = sorted(paths)
        assert len(paths) > 0, f'Expected non-empty image directory for dataset {dataset_name}!'
        
        detection_paths = glob.glob(f'datasets/{dataset_name}/vg3k_detections/*.npz')
        detection_paths += glob.glob(f'datasets/{dataset_name}/vg3k_detections/*/*.npz') # Also search subdirectories
        detection_paths = sorted(detection_paths)
        assert len(detection_paths) > 0, f'Expected non-empty directory for vg3k detections (dataset {dataset_name})!'
        assert len(detection_paths) == len(paths), f'Expected {len(paths)} detections, got {len(detection_paths)}'
        
        # Check that images are aligned
        for p, dp in zip(paths, detection_paths):
            assert os.path.basename(p).split('.')[0] == os.path.basename(dp).split('.')[0], f'Filenames do not match! ({p}, {dp})'
            
        print(f'Initialized custom dataset {dataset_name} with {len(paths)} images')
        self.paths = paths
        self.detection_paths = detection_paths
    
    def __len__(self):
        return len(self.paths)
    
    def __iter__(self):
        return zip(iter(self.paths), iter(self.detection_paths))

coco_metadata = MetadataCatalog.get('coco_2017_val')
vg3k_metadata = copy.deepcopy(coco_metadata)
vg3k_metadata.thing_classes.clear()
vg3k_metadata.thing_classes += vg3k_classes
vg3k_class_set_sorted = sorted(set(vg3k_class_set))

vg3k_name_to_idx = {}
for idx, name in enumerate(vg3k_class_set_sorted):
    indices = [i for i, x in enumerate(vg3k_metadata.thing_classes) if x == name]
    vg3k_name_to_idx[name] = idx
    

dataset_instantiators = {
    'cub': lambda: CubImageList(cmr_subset=True),
    'p3d_car': lambda: PascalImageNetList('car', cmr_subset=True),
    'p3d_airplane': lambda: PascalImageNetList('aeroplane', cmr_subset=True),
}

# Add ImageNet classes
for class_name, synsets in imagenet_synsets.items():
    dataset_instantiators['imagenet_' + class_name] = lambda synsets=synsets: ImageNetList(synsets)
    
# Add any custom dataset that may have been defined
for ds, class_name in dataset_to_class_name.items():
    if ds not in dataset_instantiators:
        dataset_instantiators[ds] = lambda ds=ds: CustomImageList(ds)


if args.datasets == 'all':
    classes = {k: v() for k, v in dataset_instantiators.items()}
elif args.datasets == 'imagenet':
    classes = {k: v() for k, v in dataset_instantiators.items() if k.startswith('imagenet_')}
elif args.datasets == 'p3d':
    classes = {k: v() for k, v in dataset_instantiators.items() if k.startswith('p3d_')}
else:
    class_filter = args.datasets.split(',')
    for cl in class_filter:
        assert cl in dataset_instantiators.keys(), f'Invalid dataset {cl}'
    classes = {k: v() for k, v in dataset_instantiators.items() if k in class_filter}

print('Selected datasets:', list(classes.keys()))
    
def initialize_predictor(disable_checks):
    global predictor

    # PointRend X101
    cfg_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
    weights_file = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0 if disable_checks else args.detection_threshold
    cfg.MODEL.WEIGHTS = weights_file
    predictor = DefaultPredictor(cfg)
    

def parse_vg3k_detections(path, threshold):
    detection = dict(np.load(path, encoding='latin1'))
    parsed_res = detection['resolution']
    vg3k_out = structures.Instances(image_size=tuple(detection['resolution']))
    vg3k_out.pred_boxes = structures.Boxes(detection['boxes'][:, :4])
    vg3k_out.pred_classes = torch.LongTensor(detection['classes'].astype(np.int))
    masks = mask_util.decode([{'counts': bytes(x), 'size': parsed_res} for x in detection['segments']])
    vg3k_out.pred_masks = torch.BoolTensor(masks).permute(2, 0, 1)
    vg3k_out.scores = torch.FloatTensor(detection['boxes'][:, 4])
    
    # Exclude detections whose confidence is below threshold
    vg3k_out = vg3k_out[vg3k_out.scores >= threshold]
    
    # Exclude detections that are part of the COCO class set
    vg3k_out = vg3k_out[vg3k_out.pred_classes > 80]
    
    # Exclude detections not in allowed list
    class_mask = [vg3k_metadata.thing_classes[x] in vg3k_class_set_sorted for x in vg3k_out.pred_classes]
    vg3k_out = vg3k_out[class_mask]
    
    return vg3k_out
                                

import torch.nn.functional as F
from collections import Counter
import random

dilation = 5
dilation_kernel = torch.ones(1, 1, dilation, dilation).cuda() 

def diff_mask(parent, child):
    assert parent.shape == child.shape
    intersection = (parent & child).sum().item()
    if intersection == 0:
        return 0
    child_area = child.sum().item()
    return intersection / child_area


with torch.no_grad():
    for dataset_name, dataset in classes.items():
        class_name = dataset_to_class_name[dataset_name]
        
        # On CUB, one image = one bird, so we can assume that there is always one instance
        disable_checks = dataset_name == 'cub'
        
        initialize_predictor(disable_checks)
        print(f'Processing dataset {dataset_name} (class {class_name}) ({len(dataset)} images)')
        results = []
        skip_count_undetected = 0
        skip_count_area = 0
        skip_count_collision = 0
        skip_count_truncated = 0
        n_instances = 0
        n_valid_instances = 0
        
        skip_paths_collision = []
        skip_paths_truncated = []
        
        instance_counter = Counter()
        for n_img, (path, detection_path) in enumerate(tqdm(dataset)):
                                
            im = cv2.imread(path)
            out = predictor(im)['instances']
            detected_class_names = [coco_metadata.thing_classes[x] for x in out.pred_classes]
            
            class_name_coco = class_name
            if class_name_coco not in detected_class_names:
                if disable_checks:
                    # Should happen rarely if the detection threshold is correctly set to 0.0
                    print(f'Warning: no {class_name} detected in {path}')
                    
                skip_count_undetected += 1
                continue
               
            relevant_indices = [i for i, name in enumerate(detected_class_names) if name == class_name_coco]
            vg3k_detections = None # Deferred loading
                
            if disable_checks:
                # Select detection with highest score
                best_prob_idx = out.scores[relevant_indices].argmax()
                relevant_indices = relevant_indices[best_prob_idx:best_prob_idx+1]
                n_instances += 1
            else:
                masks_dilated = F.conv2d(out.pred_masks.float().unsqueeze(1), dilation_kernel, padding=dilation//2)
                masks_dilated = (masks_dilated > 0).squeeze(1)

                n_instances += len(relevant_indices)
                
            
            for i in relevant_indices:
                valid = True
                
                if not disable_checks:
                    # Check for area
                    if out.pred_masks[i].sum() < 96*96:
                        skip_count_area += 1
                        continue

                    # Check for truncation (touching image borders)
                    bbox = out.pred_boxes.tensor[i].cpu()
                    margin = 10
                    w = out.pred_masks.shape[-1]
                    h = out.pred_masks.shape[-2]
                    x_min, y_min, x_max, y_max = bbox.long()
                    m = masks_dilated[i]
                    thresh = 50
                    if         (bbox[0] < margin and m[:, x_min].sum() >= thresh) \
                            or (bbox[1] < margin and m[y_min].sum() >= thresh) \
                            or (bbox[2] >= w - margin and m[:, x_max-1].sum() >= thresh) \
                            or (bbox[3] >= h - margin and m[y_max-1].sum() >= thresh):
                        skip_count_truncated += 1
                        skip_paths_truncated.append(path)
                        continue

                    # Check for collision
                    if out.pred_masks[i].float().mean() < 0.3: # Exception for main visible object
                        for j in range(len(out)):
                            if i == j:
                                continue

                            collision = (masks_dilated[i] & masks_dilated[j]).any()
                            if collision:
                                valid = False
                                break
                            if not valid:
                                break
                    if not valid:
                        skip_count_collision += 1
                        skip_paths_collision.append(path)
                        continue

                if vg3k_detections is None:
                    vg3k_detections = parse_vg3k_detections(detection_path, 0.2).to('cuda')
                    detected_class_names_vg3k = [vg3k_metadata.thing_classes[x] for x in vg3k_detections.pred_classes]
                    
                # Filter vg3k detections according to this instance
                contained_indices = []
                contained_classes = set()
                vg3k_exported_detections = []
                for j in range(len(vg3k_detections)):
                    val = diff_mask(out.pred_masks[i], vg3k_detections.pred_masks[j])
                    if val >= 0.5 and detected_class_names_vg3k[j] != class_name_coco:
                        contained_indices.append(j)
                        contained_classes.add(vg3k_detections.pred_classes[j].item())
                        
                vg3k_instance_detections = vg3k_detections[contained_indices]
                instance_counter.update(contained_classes)

                
                for j in range(len(vg3k_instance_detections)):
                    vg3k_exported_detections.append({
                        'class': vg3k_metadata.thing_classes[vg3k_instance_detections.pred_classes[j]],
                        'class_id': vg3k_name_to_idx[vg3k_metadata.thing_classes[vg3k_instance_detections.pred_classes[j]]],
                        'class_id_vg3k': vg3k_instance_detections.pred_classes[j].item(),
                        'score': vg3k_instance_detections.scores[j].item(),
                        'bbox': vg3k_instance_detections.pred_boxes[j].to('cpu').tensor.numpy(),
                        'mask': mask_util.encode(np.asfortranarray(vg3k_instance_detections.pred_masks[j].to('cpu').numpy())),
                    })
                
                # Add result
                results.append({
                    'id': n_valid_instances,
                    'class': class_name,
                    'class_id': class_indices[class_name],
                    'image_height': out.pred_masks.shape[-2],
                    'image_width': out.pred_masks.shape[-1],
                    'image_path': path,
                    'score': out.scores[i].item(),
                    'bbox': out.pred_boxes[i].to('cpu').tensor.numpy(),
                    'mask': mask_util.encode(np.asfortranarray(out.pred_masks[i].to('cpu').numpy())),
                    'num_parts': len(vg3k_exported_detections),
                    'parts': vg3k_exported_detections,
                })
                
                n_valid_instances += 1
            
        skip_count_total = skip_count_area + skip_count_collision + skip_count_truncated
        print(f'{skip_count_undetected}/{len(dataset)} images contain no detected instances')
        print(f'{skip_count_area}/{n_instances} instances skipped (small area)')
        print(f'{skip_count_collision}/{n_instances} instances skipped (collision)')
        print(f'{skip_count_truncated}/{n_instances} instances skipped (truncated)')
        print(f'{skip_count_total}/{n_instances} total instances skipped, {n_instances-skip_count_total} valid')
                 
        # Compute threshold probabilities
        prob_mapping = {}
        n_valid_instances = len(results)
        for class_idx, count in instance_counter.most_common():
            prob_mapping[class_idx] = count/n_valid_instances

        # Update all records
        for record in results:
            for part in record['parts']:
                part['frequency'] = prob_mapping[part['class_id_vg3k']]
                
        # Save
        out_dir = f'cache/{dataset_name}'
        fname = f'{out_dir}/detections'
        print(f'Saving to {fname}.npy')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.save(fname, results)
        
        print()