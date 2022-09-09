"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os.path
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from . import coco_transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, remove_no_obj_images=True):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if remove_no_obj_images:
            self.ids = sorted(list(set(
                [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())])))

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, category_map=None):
        self.return_masks = return_masks
        self.category_map = category_map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.category_map is not None:
            classes = [self.category_map[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes

        if self.category_map is None:
            target["labels"] = classes - 1
        else:
            target["labels"] = classes

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_train_transforms(train_scale_factor):
    # default
    max_size = 1333
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    random_resizes = [400, 500, 600]
    random_size_crop = (384, 600)

    if train_scale_factor != 1.0:
        # scale all with respect to custom max_size
        scales = [int(train_scale_factor * s) for s in scales]
        random_resizes = [int(train_scale_factor * s) for s in random_resizes]
        random_size_crop = [int(train_scale_factor * s) for s in random_size_crop]
        max_size = train_scale_factor * max_size

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=max_size),
            T.Compose([
                T.RandomResize(random_resizes),
                T.RandomSizeCrop(random_size_crop),
                T.RandomResize(scales, max_size=max_size),
            ])
        ),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_coco_val_transforms(val_width, max_size):
    return T.Compose([
        T.RandomResize([val_width], max_size=max_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build(image_set, cfg):
    split = cfg.get("DATASETS").get(f"{image_set}_DATASET")
    root = Path(cfg.DATASETS.DATA_PATH)

    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": ((root / "COCO/train2017", root / "COCO/annotations" / f'{mode}_train2017.json'), 91),
        "val": ((root / "COCO/val2017", root / "COCO/annotations" / f'{mode}_val2017.json'), 91),
    }
    img_folder, ann_file = PATHS[split][0]
    num_classes = PATHS[split][1]

    assert os.path.isdir(img_folder), f"Provided COCO image folder path doesn't exist {img_folder}"
    assert os.path.isfile(ann_file), f"Provided COCO annotations file doesn't exist {ann_file}"

    if image_set == "TRAIN":
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_train_transforms(cfg.INPUT.SCALE_FACTOR_TRAIN), return_masks=cfg.MODEL.MASK_ON)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_val_transforms(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST), return_masks=cfg.MODEL.MASK_ON)

    return dataset, num_classes
