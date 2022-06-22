#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:16:56 2022

@author: dinesh
"""

import json
import cv2
from PIL import Image
import numpy as np
from dateutil.parser import parse

def bb_intersection_over_union(box1, box2):
    #print(box1, box2)
    boxA = box1.copy()
    boxB = box2.copy()
    boxA[2] = boxA[0]+boxA[2]
    boxA[3] = boxA[1]+boxA[3]
    boxB[2] = boxB[0]+boxB[2]
    boxB[3] = boxB[1]+boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def bb_intersection_over_union_unoccluded(box1, box2, threshold=0.01):
    #print(box1, box2)
    boxA = box1.copy()
    boxB = box2.copy()
    boxA[2] = boxA[0]+boxA[2]
    boxA[3] = boxA[1]+boxA[3]
    boxB[2] = boxB[0]+boxB[2]
    boxB[3] = boxB[1]+boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    #print(iou)
    # return the intersection over union value
    occlusion = False
    if iou > threshold and iou < 1:
        #print(boxA[3], boxB[3], boxB[1])
        if boxA[3] < boxB[3]:# and boxA[3] > boxB[1]:
            if boxB[2] > boxA[0]:# and boxB[2] < boxA[2]:
                #print('first', (boxB[2] - boxA[0])/(boxA[2] - boxA[0]))
                if (min(boxB[2],boxA[2]) - boxA[0])/(boxA[2] - boxA[0]) > threshold:
                    occlusion = True
                
            if boxB[0] < boxA[2]: # boxB[0] > boxA[0] and 
                #print('second', (boxA[2] - boxB[0])/(boxA[2] - boxA[0]))
                if (boxA[2] - max(boxB[0],boxA[0]))/(boxA[2] - boxA[0]) > threshold:
                    occlusion = True
        if occlusion == False:
            iou = iou*0
            #asas
    #    asas
    #iou = 0.9 #iou*0
    #print(box1, box2, iou, occlusion)
    return iou
def draw_tracks(image, tracks):
    """
    Draw on input image.

    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.

    Returns:
        numpy.ndarray: image with the track-ids drawn on it.
    """

    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)

    return image


def draw_bboxes(image, tracks):
    """
    Draw the bounding boxes about detected objects in the image.

    Args:
        image (numpy.ndarray): Image or video frame.
        bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
        confidences (numpy.ndarray): Detection confidence or detection probability.
        class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.

    Returns:
        numpy.ndarray: image with the bounding boxes drawn on it.
    """

    for trk in tracks:
        xmin = int(trk[2])
        ymin = int(trk[3])
        width = int(trk[4])
        height = int(trk[5])
        clr = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), clr, 2)

    return image


def num(v):
    number_as_float = float(v)
    number_as_int = int(number_as_float)
    return number_as_int if number_as_float == number_as_int else number_as_float


def parse_bbox(bbox_str):
    bbox_list = bbox_str.strip('{').strip('}').split(',')
    bbox_list = [num(elem) for elem in bbox_list]
    return bbox_list

def parse_seg(bbox_str):
    bbox_list = bbox_str.strip('{').strip('}').split(',')
    bbox_list = [num(elem) for elem in bbox_list]
    ret = bbox_list  # []
    # for i in range(0, len(bbox_list) - 1, 2):
    #     ret.append((bbox_list[i], bbox_list[i + 1]))
    return ret
