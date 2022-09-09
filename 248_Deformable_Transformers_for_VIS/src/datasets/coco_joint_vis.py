# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# ------------------------------------------------------------------------


"""
 augment coco image to generate a n-frame pseudo video
"""
from pathlib import Path
import numpy as np
import torch
import torchvision
import os
from PIL import Image
import random

from .coco import ConvertCocoPolysToMask
from .image_to_seq_augmenter import ImageToSeqAugmenter
from . import vis_transforms as VisT
from .vis import make_train_vis_transforms
from ..util import box_ops

COCO_TO_YT19_CATEGORY_MAP = {1: 1, 2: 21, 3: 6, 4: 21, 5: 28, 7: 17, 8: 29, 9: 34, 17: 14, 18: 8,
                             19: 18, 21: 15, 22: 32, 23: 20, 24: 30, 25: 22, 36: 33, 41: 5, 42: 27,
                             43: 40
                             }

COCO_TO_YT21_CATEGORY_MAP = {1: 26, 2: 23, 3: 5, 4: 23, 5: 1, 7: 36, 8: 37, 9: 4, 16: 3, 17: 6,
                             18: 9, 19: 19, 21: 7, 22: 12, 23: 2, 24: 40, 25: 18, 36: 31, 41: 29,
                             42: 33, 43: 34, 74: 24
                             }

MAX_NUM_INSTANCES = 25


class CocoJointVIS(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder: Path, ann_file: Path, transform: VisT.VISTransformsApplier,
                 num_frames: int, num_cats: int, category_map: dict):
        super(CocoJointVIS, self).__init__(img_folder, ann_file)

        ids = []
        ids_with_objs = sorted(list(set(
            [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())])))
        for id_ in ids_with_objs:
            if len(self.coco.imgToAnns[id_]) <= MAX_NUM_INSTANCES:
                ids.append(id_)

        self.ids = ids
        self._transforms = transform
        self.num_cats = num_cats
        self.prepare = ConvertCocoPolysToMask(True, category_map)
        self.num_frames = num_frames
        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-20, 20), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5),
                                             brightness_range=(-40, 40),
                                             motion_blur_prob=0.25,
                                             motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))

    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks

    def convert_binary_masks(self, output_inst_masks):
        h, w = output_inst_masks.shape[-2:]
        num_annots = output_inst_masks.shape[1]
        areas = torch.zeros([self.num_frames, num_annots])
        segmentations = np.zeros((self.num_frames, h, w, 1), dtype=np.uint8)

        for j in range(self.num_frames):
            for i in range(num_annots):
                instance_mask = output_inst_masks[j, i]
                segmentations[j, instance_mask.astype(np.bool_)] = i + 1
                areas[j, i] = instance_mask.sum()

        return segmentations, areas

    def __getitem__(self, idx):

        img, target = super(CocoJointVIS, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        seq_images, seq_instance_masks = [img], [target['masks'].numpy()]
        numpy_masks = target['masks'].numpy()

        num_inst = len(numpy_masks)
        for t in range(self.num_frames - 1):
            im_trafo, instance_masks_trafo = self.augmenter(np.asarray(img), numpy_masks)
            im_trafo = Image.fromarray(np.uint8(im_trafo))
            seq_images.append(im_trafo)
            seq_instance_masks.append(np.stack(instance_masks_trafo, axis=0))
        seq_images, seq_instance_masks = self.apply_random_sequence_shuffle(seq_images,
                                                                            seq_instance_masks)
        output_inst_masks = []
        for inst_i in range(num_inst):
            inst_i_mask = []
            for f_i in range(self.num_frames):
                inst_i_mask.append(seq_instance_masks[f_i][inst_i])
            output_inst_masks.append(np.stack(inst_i_mask, axis=0))

        output_inst_masks = np.stack(output_inst_masks, axis=0).swapaxes(0, 1)
        valued_masks, areas = self.convert_binary_masks(output_inst_masks)
        boxes = box_ops.masks_to_boxes(torch.from_numpy(output_inst_masks).flatten(0, 1))
        boxes = boxes.unflatten(0, (self.num_frames, num_inst))

        target['masks'] = valued_masks
        target['boxes'] = boxes.numpy()
        target['labels'] = target['labels'].repeat(self.num_frames, 1).numpy()
        target['valid'] = torch.ones([self.num_frames, num_inst])
        target['area'] = areas
        seq_images = [np.array(image) for image in seq_images]

        if self._transforms is not None:
            img, target = self._transforms(seq_images, target)

        target["labels"] = target["labels"] - 1
        # Background is set to last logit
        for idx in range(target["labels"].shape[0]):
            if target["labels"][idx] == -1:
                target["labels"][idx] = self.num_cats

        if isinstance(img, list):
            img = torch.cat(img, dim=0)

        return img, target


def build_coco_joint_vis(cfg, num_cats):
    root = Path(cfg.DATASETS.DATA_PATH)
    assert root.exists(), f'provided Data path {root} does not exist'

    split = cfg.get("DATASETS").get("TRAIN_DATASET")
    img_folder = root / "COCO/train2017"
    ann_file = root / "COCO/annotations" / f'coco_keepfor_ytvis19.json'

    assert os.path.isdir(img_folder), f"Provided VIS image folder path doesn't exist {img_folder}"
    assert os.path.isfile(ann_file), f"Provided VIS annotations file doesn't exist {ann_file}"

    CATEGORY_MAP = {
        "yt_vis_train_19": COCO_TO_YT19_CATEGORY_MAP,
        "yt_vis_train_21": COCO_TO_YT21_CATEGORY_MAP,
        "mini_train": COCO_TO_YT19_CATEGORY_MAP,

    }

    cat_map = CATEGORY_MAP[split]
    transforms = make_train_vis_transforms(cfg.INPUT.SCALE_FACTOR_TRAIN,
                                           cfg.INPUT.DEVIS.MULTI_SCALE_TRAIN,
                                           cfg.INPUT.DEVIS.CREATE_BBX_FROM_MASK)

    # Delete photometric augmentation from the default VIS pipeline, leave rest unchanged
    transforms.remove_transform(VisT.VISPhotometricDistort)

    dataset = CocoJointVIS(img_folder, ann_file, transform=transforms,
                           num_frames=cfg.MODEL.DEVIS.NUM_FRAMES, num_cats=num_cats,
                           category_map=cat_map)

    return dataset
