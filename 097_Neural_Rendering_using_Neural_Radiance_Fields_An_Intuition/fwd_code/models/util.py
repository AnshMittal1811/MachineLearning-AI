import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_black(input_img, threshold = -0.95):
    black_detect = torch.bitwise_and(input_img[:,0:1]<=threshold, torch.bitwise_and(input_img[:, 1:2]<= threshold, input_img[:, 2:3]<= threshold))
    return black_detect

def detect_background(depth_img, input_img, threshold=-0.95):

    black_detect = torch.bitwise_and(input_img[:,0:1]<=threshold, torch.bitwise_and(input_img[:, 1:2]<= threshold, input_img[:, 2:3]<= threshold))

    unknow_depth = depth_img == 0

    return torch.bitwise_and(unknow_depth, black_detect)