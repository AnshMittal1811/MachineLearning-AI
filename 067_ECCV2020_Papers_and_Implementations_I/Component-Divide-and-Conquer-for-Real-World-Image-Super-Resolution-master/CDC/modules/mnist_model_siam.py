# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from stn.grid_sample import grid_sample
from torch.autograd import Variable
from stn.tps_grid_gen import TPSGridGen


class CNN_siam(nn.Module):
    def __init__(self, num_output, scale):
        super(CNN_siam, self).__init__()

        self.scale = scale
        self.ups = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

        self.scale = scale

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        input1_ups = self.ups(input1)
        output1 = self.forward_once(input1_ups)
        output2 = self.forward_once(input2)
        res = output1 - output2
        x = res.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return output1, output2
        return x, input1_ups


class CNN_siam_l(nn.Module):
    def __init__(self, num_output, scale):
        super(CNN_siam_l, self).__init__()

        self.scale = scale
        self.ups = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

        self.scale = scale

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        input1_ups = self.ups(input1)
        output1 = self.forward_once(input1_ups)
        output2 = self.forward_once(input2)
        res = output1 - output2
        x = res.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return output1, output2
        return x, input1_ups


class CNN_siam_ll(nn.Module):
    """
    Transform parameter generate model
    Input: lr, hr
    Output: Transform matrix
    """
    def __init__(self, num_output, scale):
        super(CNN_siam_ll, self).__init__()

        self.scale = scale
        self.ups = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4_drop = nn.Dropout2d()

        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

        self.scale = scale

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        input1_ups = self.ups(input1)
        output1 = self.forward_once(input1_ups)
        output2 = self.forward_once(input2)
        res = output1 - output2
        res = self.pool(res)
        x = res.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return output1, output2
        return x, input1_ups


class CNN_siam_sr(nn.Module):
    """
    Transform parameter generate model
    Input: SR (upsampled from LR by pretrained sr model), HR
    Output: Transform matrix
    """
    def __init__(self, num_output, scale):
        super(CNN_siam_sr, self).__init__()

        self.scale = scale
        # self.ups = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4_drop = nn.Dropout2d()

        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

        self.scale = scale

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return x

    def forward(self, input1_ups, input2):
        # input1_ups = self.ups(input1)
        output1 = self.forward_once(input1_ups)
        output2 = self.forward_once(input2)
        res = output1 - output2
        res = self.pool(res)
        x = res.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return output1, output2
        return x, input1_ups


class BoundedGridLocNet_sr_warp(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, scale):
        super(BoundedGridLocNet_sr_warp, self).__init__()
        self.cnn = CNN_siam(grid_height * grid_width * 2, scale)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        output, input_ups = self.cnn(x1, x2)
        points = F.tanh(output)
        return points.view(batch_size, -1, 2), input_ups


class UnBoundedGridLocNet_sr_warp(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, scale, sr_label=False):
        super(UnBoundedGridLocNet_sr_warp, self).__init__()
        if sr_label:
            self.cnn = CNN_siam_sr(grid_height * grid_width * 2, scale)
        else:
            self.cnn = CNN_siam_ll(grid_height * grid_width * 2, scale)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        points, input_ups = self.cnn(x1, x2)
        return points.view(batch_size, -1, 2), input_ups


class STNClsNet_sr_warp(nn.Module):

    def __init__(self, args):
        super(STNClsNet_sr_warp, self).__init__()
        self.args = args

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1    # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet_sr_warp,
            'bounded_stn': BoundedGridLocNet_sr_warp,
        }[args.stn_model]
        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points, args.scale[0], args.sr_label)

        self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points)

        #self.cls_net = ClsNet()

    def forward(self, x1, x2):
        """
        :param x1: reference(target), lr
        :param x2: source, to be changed, hr
        :return:
        """
        batch_size = x1.size(0)
        source_control_points, input_ups = self.loc_net(x1, x2)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.args.image_height, self.args.image_width, 2)
        transformed_x = grid_sample(x2, grid)
        #logit = self.cls_net(transformed_x)
        return transformed_x, input_ups


def get_model(args):
    if args.stn_model == 'no_stn':
        print('create model without STN')
        model = CNN_siam()
    else:
        print('create model with STN')
        model = STNClsNet_sr_warp(args)
    return model