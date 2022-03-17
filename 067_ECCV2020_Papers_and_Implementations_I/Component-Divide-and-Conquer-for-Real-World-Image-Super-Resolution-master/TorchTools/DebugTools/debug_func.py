import numpy as np
import torch
import cv2

def show_img_from_tensor(tensor, save_dir='', show=False):
    """
    visualize tensor / Variable
    :param tensor: image
    :param show:
    :param save_dir: path to save image
    :return:
    """
    im = torch.squeeze(tensor) * 255
    im = im.numpy().clip(0, 255)
    im = im.astype(np.uint8)
    if show:
        cv2.imshow('test', im)
        cv2.waitKey(0)
    if save_dir != '':
        cv2.imwrite(save_dir, im)
    return im
