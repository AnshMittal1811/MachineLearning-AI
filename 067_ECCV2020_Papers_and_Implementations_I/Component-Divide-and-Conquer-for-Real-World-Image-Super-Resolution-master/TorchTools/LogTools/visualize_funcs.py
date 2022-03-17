import os
import pdb
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import random
import re
from torchvision import transforms
import torch
import cv2

def plot_loss(loss, label, im_path):
    x_loss = [i + 1 for i in range(len(loss))]
    linewidth = 0.75
    if label == 'train':
        color = [0 / 255.0, 191 / 255.0, 255 / 255.0]
    else:
        color = [255 / 255.0, 106 / 255.0, 106 / 255.0]
    title = label + '_loss'
    plt.plot(x_loss, loss, color=color, linewidth=linewidth)
    plt.title(title)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(im_path))
    plt.show()

def draw_log(log_path, save_path, col_num, caffe_log=False, train=True):
    """
    draw log from log.txt
    :param log_path:
    :param save_path: folder name
    :param col_num: colume number of loss
    :param caffe_log:
    :param train: train loss / test loss
    :return:
    """
    log_file = open(log_path, 'r')
    im_name = log_path.split('/')[-1].split('.')[0] + '.png'
    train_loss = []
    test_loss = []
    lines = log_file.readlines()
    cnt = -1
    for line in lines:
        cnt += 1
        if cnt % 2 == 0:
            continue
        line = line.strip().split(' ')
        # delete_space(line)
        if caffe_log:
            if len(line) > 4 and line[4] == 'Train':
                loss = float(line[col_num])
                train_loss.append(loss)
            elif len(line) > 4 and line[4] == 'Test':
                loss = float(line[col_num])
                test_loss.append(loss)
        else:
            loss = float(line[col_num].strip(','))
            train_loss.append(loss)
    save_path = os.path.join(save_path, im_name)
    plot_loss(train_loss, 'train' if train else 'test', save_path)

def show_img_from_tensor(tensor, save_dir='', show=False):
    """
    visualize tensor / Variable
    :param tensor: image
    :param show:
    :param save_dir: path to save image
    :return:
    """
    tensor2im = transforms.ToPILImage()
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, 0)
    im = tensor2im(tensor)
    if show:
        im.show()
    if save_dir != '':
        im.save(save_dir)
    # im = torch.squeeze(tensor) * 255
    # im = im.numpy().clip(0, 255)
    # im = im.astype(np.uint8)
    # if show:
    #     cv2.imshow('test', im)
    #     cv2.waitKey(0)
    # if save_dir != '':
    #     cv2.imwrite(save_dir, im)
    # return im


