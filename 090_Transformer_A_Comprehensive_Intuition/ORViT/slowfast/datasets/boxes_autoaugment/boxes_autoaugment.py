#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified

import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np
from . import bbox_util as bbox_aug

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage

RAD_TO_DEG = 57.2958

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])
assert _PIL_VER >= (5, 2)

def identity(img, *factor, **kwargs):
    return img

def rotate(boxes, factor, **kwargs):
    return bbox_aug.rotate_boxes(boxes, factor, size_before=kwargs.pop('size_before'))

def shear_x(boxes, factor, **kwargs):
    if factor[0] == 0: return boxes
    return bbox_aug.shearX_boxes(boxes, factor, size_before=kwargs.pop('size_before'))

def shear_x_iaa(boxes, factor, **kwargs):
    if factor == 0: return boxes
    factor = factor * RAD_TO_DEG
    w, h= kwargs.pop('size_before')
    iboxes = BoundingBoxesOnImage([BoundingBox(*b.tolist()) for b in boxes], (h,w))
    seq = iaa.geometric.ShearX(factor, fit_output=True).to_deterministic()
    boxes = seq.augment_bounding_boxes([iboxes])[0]
    w_after, h_after = boxes.width, boxes.height
    boxes = boxes.to_xyxy_array()

    # scale height
    h_scale = h / h_after
    boxes[:, [1,3]] = boxes[:, [1,3]] * h_scale

    # crop width
    w_start = w_after - w
    if factor > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - w_start
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    return boxes


def shear_x_iaa_img(img, factor, **kwargs):
    if factor == 0: return img
    factor = factor*RAD_TO_DEG
    w, h= img.size
    img = np.array(img)
    seq = iaa.geometric.ShearX(factor, fit_output=True).to_deterministic()
    seq2 = iaa.size.Resize({'height':h})
    img = seq(images=[img])[0]
    img = seq2(images=[img])[0]
    h_after ,w_after = img.shape[0], img.shape[1]
    if factor > 0:
        img = img[:, -w:] # crop w
    else:
        img = img[:, :w] # crop w

    img = Image.fromarray(img.astype(np.uint8))

    return img


def shear_y_iaa(boxes, factor, **kwargs):
    if factor == 0: return boxes
    factor = -factor * RAD_TO_DEG
    w, h= kwargs.pop('size_before')
    iboxes = BoundingBoxesOnImage([BoundingBox(*b.tolist()) for b in boxes], (h,w))
    seq = iaa.geometric.ShearY(factor, fit_output=True).to_deterministic()
    boxes = seq.augment_bounding_boxes([iboxes])[0]
    w_after, h_after = boxes.width, boxes.height
    boxes = boxes.to_xyxy_array()

    # scale width
    w_scale = w / w_after
    boxes[:, [0,2]] = boxes[:, [0,2]] * w_scale

    # crop height
    if factor < 0:
        h_start = h_after - h
        boxes[:, [1,3]] = boxes[:, [1,3]] - h_start
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, h)

    return boxes


def shear_y_iaa_img(img, factor, **kwargs):
    if factor == 0: return img
    factor = -factor*RAD_TO_DEG
    w, h= img.size
    img = np.array(img)
    seq = iaa.geometric.ShearY(factor, fit_output=True).to_deterministic()
    seq2 = iaa.size.Resize({'width':w})
    img = seq(images=[img])[0]
    img = seq2(images=[img])[0]
    h_after ,w_after = img.shape[0], img.shape[1]
    if factor < 0:
        img = img[-h:] # crop h
    else:
        img = img[:h]
    img = Image.fromarray(img.astype(np.uint8))

    return img


def translate_x_rel(boxes, factor, **kwargs):
    w, h = kwargs.pop('size_before')
    offset = w * factor
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset
    return boxes


def translate_y_rel(boxes, factor, **kwargs):
    w, h = kwargs.pop('size_before')
    offset = h * factor
    boxes = boxes.copy()
    boxes[:, [1,3]] = boxes[:, [1,3]] - offset
    return boxes

def not_implemented(boxes, factor, **kwargs):
    raise NotImplementedError()




NAME_TO_OP = {
    'AutoContrast': identity,
    'Equalize': identity,
    'Invert': identity, 
    'Rotate': rotate, # DONE
    'Posterize': identity,
    'PosterizeIncreasing': identity,
    'PosterizeOriginal': identity,
    'Solarize': identity,
    'SolarizeIncreasing': identity,
    'SolarizeAdd': identity,
    'Color': identity,
    'ColorIncreasing': identity,
    'Contrast': identity,
    'ContrastIncreasing': identity,
    'Brightness': identity,
    'BrightnessIncreasing': identity,
    'Sharpness': identity,
    'SharpnessIncreasing': identity,
    'ShearX': shear_x_iaa, ## DONE
    'ShearY': shear_y_iaa, ## DONE
    'TranslateX': not_implemented, ##
    'TranslateY': not_implemented, ##
    'TranslateXRel': translate_x_rel, ## DONE
    'TranslateYRel': translate_y_rel, ## DONE
}


##
