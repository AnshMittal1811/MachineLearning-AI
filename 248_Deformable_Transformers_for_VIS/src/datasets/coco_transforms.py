# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""

import random
from typing import Union
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

import numpy as np
from numpy import random as rand
from PIL import Image
import cv2

from ..util.box_ops import box_xyxy_to_cxcywh
from ..util.misc import interpolate


def crop(image, target, region):
    i, j, h, w = region
    target = target.copy()

    if isinstance(image, torch.Tensor):
        cropped_image = image[:, j:j + w, i:i + h]
    else:
        cropped_image = F.crop(image, *region)

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "ignore", "track_ids"]

    orig_area = target["area"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "masks" in target:
            keep = target['masks'].flatten(1).any(1)
        else:
            # cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            # keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

            # new area must be at least % of orginal area
            keep = target["area"] >= orig_area * 0.2

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    if isinstance(image, torch.Tensor):
        flipped_image = image.flip(-1)
        _, width, _ = image.size()
    else:
        flipped_image = F.hflip(image)
        width, _ = image.size

    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] \
                * torch.as_tensor([-1, 1, -1, 1]) \
                + torch.as_tensor([width, 0, width, 0])
        target["boxes"] = boxes

    if "boxes_ignore" in target:
        boxes = target["boxes_ignore"]
        boxes = boxes[:, [2, 1, 0, 3]] \
                * torch.as_tensor([-1, 1, -1, 1]) \
                + torch.as_tensor([width, 0, width, 0])
        target["boxes_ignore"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


def get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes \
                       * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # pad_left, pad_top, pad_right, pad_bottom
    padded_image = F.pad(image, padding)
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    w, h = padded_image.size

    if "boxes" in target:
        # correct xyxy from left and right paddings
        target["boxes"] += torch.tensor(
            [padding[0], padding[1], padding[0], padding[1]])

    target["size"] = torch.tensor([h, w])
    if "masks" in target:
        # padding_left, padding_right, padding_top, padding_bottom
        target['masks'] = torch.nn.functional.pad(
            target['masks'],
            (padding[0], padding[2], padding[1], padding[3]))
    return padded_image, target


class RandomCrop:
    def __init__(self, size):
        # in hxw
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop:
    def __init__(self,
                 min_size: Union[tuple, list, int],
                 max_size: Union[tuple, list, int] = None):
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        if isinstance(max_size, int):
            max_size = (max_size, max_size)

        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        if self.max_size is None:
            w = random.randint(min(self.min_size[0], img.width), img.width)
            h = random.randint(min(self.min_size[1], img.height), img.height)
        else:
            w = random.randint(
                min(self.min_size[0], img.width),
                min(img.width, self.max_size[0]))
            h = random.randint(
                min(self.min_size[1], img.height),
                min(img.height, self.max_size[1]))

        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CustomRandomSizeCrop:
    def __init__(self,
                 min_size: Union[tuple, list, int],
                 max_size: Union[tuple, list, int] = None):
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        if isinstance(max_size, int):
            max_size = (max_size, max_size)

        self.min_size = min_size
        self.max_size = max_size
        self.max_calls = 5

    def get_w_h(self, img):
        if self.max_size is None:
            w = random.randint(min(self.min_size[0], img.width), img.width)
            h = random.randint(min(self.min_size[1], img.height), img.height)
        else:
            w = random.randint(
                min(self.min_size[0], img.width),
                min(img.width, self.max_size[0]))
            h = random.randint(
                min(self.min_size[1], img.height),
                min(img.height, self.max_size[1]))

        return h, w

    def __call__(self, img: PIL.Image.Image, target: dict):
        valid = False
        count = 0
        cropped_image, cropped_target = None, None
        while not valid and count < self.max_calls:
            h, w = self.get_w_h(img)
            region = T.RandomCrop.get_params(img, [h, w])
            cropped_image, cropped_target = crop(img, target, region)
            if not cropped_target["boxes"].shape[0] == 0:
                valid = True
            else:
                count += 1

        if count >= self.max_calls:
            cropped_image, cropped_target = img, target
        return cropped_image, cropped_target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target


class RandomHue(object):  #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, target):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target


# NOTICE: if used for mask, need to change
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, clip, target):
        if rand.randint(2):
            return clip, target
        imgs = []
        masks = []
        image = np.asarray(clip[0]).astype('float32')
        height, width, depth = image.shape
        ratio = rand.uniform(1, 4)
        left = rand.uniform(0, width * ratio - width)
        top = rand.uniform(0, height * ratio - height)
        for i in range(len(clip)):
            image = np.asarray(clip[i]).astype('float32')
            expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height), int(left):int(left + width)] = image
            imgs.append(Image.fromarray(expand_image.astype('uint8')))
            expand_mask = torch.zeros((int(height * ratio), int(width * ratio)), dtype=torch.uint8)
            expand_mask[int(top):int(top + height), int(left):int(left + width)] = target['masks'][i]
            masks.append(expand_mask)
        boxes = target['boxes'].numpy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = torch.tensor(boxes)
        target['masks'] = torch.stack(masks)
        return imgs, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RepeatUntilMaxObjects:
    def __init__(self, transforms, num_max_objects):
        self._num_max_objects = num_max_objects
        self._transforms = transforms

    def __call__(self, img, target):
        num_objects = None
        while num_objects is None or num_objects > self._num_max_objects:
            img_trans, target_trans = self._transforms(img, target)
            num_objects = len(target_trans['boxes'])
        return img_trans, target_trans


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomResizeTargets:
    def __init__(self, scale=0.5):
        self.scalce = scale

    def __call__(self, img, target=None):
        img = F.to_tensor(img)
        img_c, img_w, img_h = img.shape

        rescaled_boxes = []
        rescaled_box_images = []
        for box in target['boxes']:
            y1, x1, y2, x2 = box.int().tolist()
            w = x2 - x1
            h = y2 - y1

            box_img = img[:, x1:x2, y1:y2]
            random_scale = random.uniform(0.5, 2.0)
            scaled_width = int(random_scale * w)
            scaled_height = int(random_scale * h)

            box_img = F.to_pil_image(box_img)
            rescaled_box_image = F.resize(
                box_img,
                (scaled_width, scaled_height))
            rescaled_box_images.append(F.to_tensor(rescaled_box_image))
            rescaled_boxes.append([y1, x1, y1 + scaled_height, x1 + scaled_width])

        for box in target['boxes']:
            y1, x1, y2, x2 = box.int().tolist()
            w = x2 - x1
            h = y2 - y1

            erase_value = torch.empty(
                [img_c, w, h],
                dtype=torch.float32).normal_()

            img = F.erase(
                img, x1, y1, w, h, erase_value, True)

        for box, rescaled_box_image in zip(target['boxes'], rescaled_box_images):
            y1, x1, y2, x2 = box.int().tolist()
            w = x2 - x1
            h = y2 - y1
            _, scaled_width, scaled_height = rescaled_box_image.shape

            rescaled_box_image = rescaled_box_image[
                                 :,
                                 :scaled_width - max(x1 + scaled_width - img_w, 0),
                                 :scaled_height - max(y1 + scaled_height - img_h, 0)]

            img[:, x1:x1 + scaled_width, y1:y1 + scaled_height] = rescaled_box_image

        target['boxes'] = torch.tensor(rescaled_boxes).float()
        img = F.to_pil_image(img)
        return img, target


class RandomPad:
    def __init__(self, max_size):
        if isinstance(max_size, int):
            max_size = (max_size, max_size)

        self.max_size = max_size

    def __call__(self, img, target):
        w, h = img.size
        pad_width = random.randint(0, max(self.max_size[0] - w, 0))
        pad_height = random.randint(0, max(self.max_size[1] - h, 0))

        pad_left = random.randint(0, pad_width)
        pad_right = pad_width - pad_left
        pad_top = random.randint(0, pad_height)
        pad_bottom = pad_height - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return pad(img, target, padding)


class RandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(self, img, target=None):
        return F.to_tensor(img), target


class RandomErasing:

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.eraser = T.RandomErasing()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img, target):
        if random.uniform(0, 1) < self.p:
            img = F.to_tensor(img)

            x, y, h, w, v = self.eraser.get_params(
                img, scale=self.scale, ratio=self.ratio, value=self.value)

            img = F.erase(img, x, y, h, w, v, self.inplace)
            img = F.to_pil_image(img)

            # target
            fields = ['boxes', "labels", "area", "iscrowd", "ignore", "track_ids"]

            if 'boxes' in target:
                erased_box = torch.tensor([[y, x, y + w, x + h]]).float()

                lt = torch.max(erased_box[:, None, :2], target['boxes'][:, :2])  # [N,M,2]
                rb = torch.min(erased_box[:, None, 2:], target['boxes'][:, 2:])  # [N,M,2]
                wh = (rb - lt).clamp(min=0)  # [N,M,2]
                inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

                keep = inter[0] <= 0.7 * target['area']

                left = torch.logical_and(
                    target['boxes'][:, 0] < erased_box[:, 0],
                    target['boxes'][:, 2] > erased_box[:, 0])
                left = torch.logical_and(left, inter[0].bool())

                right = torch.logical_and(
                    target['boxes'][:, 0] < erased_box[:, 2],
                    target['boxes'][:, 2] > erased_box[:, 2])
                right = torch.logical_and(right, inter[0].bool())

                top = torch.logical_and(
                    target['boxes'][:, 1] < erased_box[:, 1],
                    target['boxes'][:, 3] > erased_box[:, 1])
                top = torch.logical_and(top, inter[0].bool())

                bottom = torch.logical_and(
                    target['boxes'][:, 1] < erased_box[:, 3],
                    target['boxes'][:, 3] > erased_box[:, 3])
                bottom = torch.logical_and(bottom, inter[0].bool())

                only_one_crop = (top.float() + bottom.float() + left.float() + right.float()) > 1
                left[only_one_crop] = False
                right[only_one_crop] = False
                top[only_one_crop] = False
                bottom[only_one_crop] = False

                target['boxes'][:, 2][left] = erased_box[:, 0]
                target['boxes'][:, 0][right] = erased_box[:, 2]
                target['boxes'][:, 3][top] = erased_box[:, 1]
                target['boxes'][:, 1][bottom] = erased_box[:, 3]

                for field in fields:
                    if field in target:
                        target[field] = target[field][keep]

        return img, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
