import itertools
import os
import random

from PIL import Image, ImageChops
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

from .OpticalFlow import Advector, Warp, FlowField
from .modules import upsampleBlock
from .Losses import TVLoss
from ..DataTools.Prepro import _id, _tanh_to_sigmoid, _sigmoid_to_tanh, random_pre_process
from ..DataTools.FileTools import _image_file
from ..DataTools.Loaders import pil_loader, PIL2VAR, VAR2PIL, _add_batch_one
from ..Functions import functional as Func


class MapsAdvector(nn.Module):
    def __init__(self, input_channel=64, freedom_degree=2):
        super(MapsAdvector, self).__init__()
        self.conv1 = nn.Conv2d(input_channel * 2, 32, 5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.linear = nn.Linear(32, freedom_degree)

    def forward(self, frame_t, frame_tp1):
        input = torch.cat([frame_t, frame_tp1], dim=1)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = self.linear(x.view(x.size()[:2]))
        return x


class MapsFlow(nn.Module):
    def __init__(self, input_channel=64):
        super(MapsFlow, self).__init__()
        self.conv1 = nn.Conv2d(input_channel * 2, 32, 5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.pixel1 = nn.PixelShuffle(4)

    def forward(self, frame_t, frame_tp1):
        input = torch.cat([frame_t, frame_tp1], dim=1)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.tanh(self.pixel1(x))
        return x


class RenewNet(nn.Module):
    def __init__(self, channels=64):
        super(RenewNet, self).__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)

    def forward(self, input, memory):
        x = torch.cat([input, memory], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class SumWeight(nn.Module):
    def __init__(self, channels=64):
        super(SumWeight, self).__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)

    def forward(self, input, memory):
        x = torch.cat([input, memory], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x


class FrameWeight(nn.Module):
    def __init__(self, in_channels=1, channels=64):
        super(FrameWeight, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, channels, 5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, in_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, input_list, ref_frame):
        x = torch.cat([input_list, ref_frame], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x


class MapsWarp(nn.Module):
    def __init__(self, channels=64):
        super(MapsWarp, self).__init__()
        self.advector = MapsAdvector(input_channel=channels)
        self.warp = Advector()

    def forward(self, input, memory):
        advector = self.advector(input, memory)
        return self.warp(input, advector)


class SRConvlayer(nn.Module):
    def __init__(self, layers=7, input_channels=1, channels=64):
        super(SRConvlayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, channels, 5, stride=1, padding=2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.layers = layers
        for i in range(layers):
            self.add_module(
                'conv' + str(i + 1),
                nn.Sequential(*[
                    nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            )

    def forward(self, input):
        x = self.relu(self.conv(input))
        y = x
        for i in range(self.layers):
            x = self.__getattr__('conv' + str(i + 1))(x)
        return torch.add(x, y)


class SRUpsampleBlock(nn.Module):
    def __init__(self, scala=4, input_channels=64, output_channels=1):
        super(SRUpsampleBlock, self).__init__()
        self.scala = scala
        for i in range(int(log2(scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(input_channels, input_channels * 4, activation=nn.LeakyReLU(0.2, inplace=True)))
        self.convf = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, input):
        x = input
        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)
        return F.tanh(self.convf(x))


class BurstSRDataSet(data.Dataset):
    """
    DataSet for small images, easy to read
    do not need buffer
    random crop.
    all the image are same size
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 burst_number=16,
                 scala=4,
                 misalignment=2,
                 misalignment_shift=0,
                 poisson_lambda=1.0,
                 noise=True,
                 tensor=False,
                 interp=Image.BICUBIC,
                 mode='Y',
                 prepro=random_pre_process):
        """
            :param data_path: Path to data root
            :param lr_patch_size: the Low resolution size, by default, the patch is square
            :param scala: SR scala, default is 4
            :param interp: interpolation for resize, default is Image.BICUBIC, optional [Image.BILINEAR, Image.BICUBIC]
            :param mode: 'RGB' or 'Y'
            :param sub_dir: if True, then all the images in the `data_path` directory AND child directory will be use
            :parem prepro: function fo to ``PIL.Image``!, will run this function before crop and resize
        """
        self.channels = 1 if mode == 'Y' else 3
        self.noise = noise
        self.burst = burst_number
        self.uniform = lambda: int(np.random.uniform(low=-misalignment * scala, high=misalignment * scala))
        self.shift = lambda: True if np.random.uniform(low=0, high=1) < (np.random.poisson(poisson_lambda) / burst_number) else False
        self.shift_uniform = lambda: int(np.random.uniform(low=-misalignment_shift, high=misalignment_shift))
        data_path = os.path.abspath(data_path)
        print('Initializing DataSet, data root: %s ...' % data_path)
        self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.misalignment = misalignment
        self.cropable_size = self.crop_size + misalignment * scala * 2
        self.prepro = prepro
        self.is_tensor = tensor

    def gt_crop(self, img):
        img = self.prepro(img)
        w, h = img.size
        w_start = random.randint(0, w - self.cropable_size)
        h_start = random.randint(0, h - self.cropable_size)
        gt_full = Func.crop(img, h_start + self.uniform(), w_start + self.uniform(), self.cropable_size, self.cropable_size)
        croped_gt = Func.crop(gt_full, self.misalignment * self.scala, self.misalignment * self.scala, self.crop_size, self.crop_size)
        return croped_gt, gt_full

    def _to_tensor(self, burst_list):
        if self.is_tensor:
            gt = burst_list.pop(0)
            burst = torch.cat(burst_list, dim=0)
            return burst, gt
        else:
            return burst_list


    def _burst_prepro_crop(self, img):
        w_start = self.misalignment * self.scala + self.uniform()
        h_start = self.misalignment * self.scala + self.uniform()
        croped = Func.crop(img, h_start, w_start, self.crop_size, self.crop_size)
        return croped

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        if self.mode == 'Y':
            image = pil_loader(self.image_file_list[index], mode='YCbCr')
        else:
            image = pil_loader(self.image_file_list[index], mode=self.mode)
        return_list = [None] * (self.burst + 1)
        hr_gt, gt_full = self.gt_crop(self.prepro(image))
        return_list[1] = Func.to_tensor(Func.resize(hr_gt, self.lr_size))[:self.channels]
        for i in range(2, self.burst + 1):
            lr_burst = Func.resize(self._burst_prepro_crop(gt_full), self.lr_size)
            if self.shift():
                lr_burst = ImageChops.offset(lr_burst, self.shift_uniform(), self.shift_uniform())
                return_list[i] = Func.to_tensor(lr_burst)[:self.channels]
            else:
                return_list[i] = Func.to_tensor(lr_burst)[:self.channels]
        return_list[0] = Func.to_tensor(hr_gt)[:self.channels]
        return self._to_tensor(return_list)

    def __len__(self):
        return len(self.image_file_list)


class BurstSR(nn.Module):

    def __init__(self, scala=4):
        super(BurstSR, self).__init__()
        self.warp = Warp()
        self.flow = FlowField()
        self.tvloss = TVLoss()
        self.srconv = SRConvlayer()
        self.upsample = SRUpsampleBlock(scala=scala)
        self.renew = RenewNet()

    def init_zoo(self, srconv='', upsample='', renew='', flow=''):
        if srconv != '':
            print('Loading srconv: %s' % srconv)
            self.srconv.load_state_dict(torch.load(srconv))
        if upsample != '':
            print('Loading upsample: %s' % upsample)
            self.upsample.load_state_dict(torch.load(upsample))
        if renew != '':
            print('Loading renew: %s' % renew)
            self.renew.load_state_dict(torch.load(renew))
        if flow != '':
            print('Loading flow: %s' % flow)
            self.flow.load_state_dict(torch.load(flow))

    def forward(self, frame, frame_ref=None, memory=None):
        if memory is None and frame_ref is None:
            new_memory = self.srconv(frame)
            sr = self.upsample(new_memory)
            return sr, new_memory
        else:
            flow = self.flow(frame, frame_ref)
            warped = self.warp(frame, flow)

            new_memory = self.srconv(warped)

            renew = self.renew(memory, new_memory)
            renew_memory = torch.add(renew, memory)
            sr = self.upsample(renew_memory)
            return sr, renew_memory, flow, warped


class BurstSRWeightedSum(nn.Module):

    def __init__(self, scala=4):
        super(BurstSRWeightedSum, self).__init__()
        self.srconv = SRConvlayer()
        self.upsample = SRUpsampleBlock(scala=scala)
        self.renew = SumWeight()

    def init_zoo(self, srconv='', upsample='', renew='', flow=''):
        if srconv != '':
            print('Loading srconv: %s' % srconv)
            self.srconv.load_state_dict(torch.load(srconv))
        if upsample != '':
            print('Loading upsample: %s' % upsample)
            self.upsample.load_state_dict(torch.load(upsample))
        if renew != '':
            print('Loading renew: %s' % renew)
            self.renew.load_state_dict(torch.load(renew))

    def forward(self, frame, frame_ref=None, memory=None):
        if memory is None and frame_ref is None:
            new_memory = self.srconv(frame)
            sr = self.upsample(new_memory)
            return sr, new_memory
        else:
            new_memory = self.srconv(frame)

            renew = self.renew(memory, new_memory)
            renew_memory = renew * new_memory + (1 - renew) * memory
            sr = self.upsample(renew_memory)
            return sr, renew_memory


class _FrameWeighted(nn.Module):
    def __init__(self):
        super(_FrameWeighted, self).__init__()
        self.renew = FrameWeight()

    def init_zoo(self, renew=''):
        if renew != '':
            print('Loading renew: %s' % renew)
            self.renew.load_state_dict(torch.load(renew))

    def forward(self, input_list):
        ref = input_list[0]
        length = len(input_list)
        fusion_list = list()
        for i in range(1, length):
            weight = self.renew(input_list[i], ref)
            i_fusion = weight * ref + (1 - weight) * input_list[i]
            fusion_list.append(i_fusion)
        fusion = torch.div(sum(fusion_list), length)
        return fusion


class _FrameWeightedAvg(nn.Module):
    def __init__(self):
        super(_FrameWeightedAvg, self).__init__()
        self.renew = FrameWeight()

    def init_zoo(self, renew=''):
        if renew != '':
            print('Loading renew: %s' % renew)
            self.renew.load_state_dict(torch.load(renew))

    def forward(self, input_list):
        ref = input_list[0]
        length = len(input_list)
        fusion_list = list()
        for i in range(1, length):
            weight = self.renew(input_list[i], ref) / 2.
            i_fusion = (0.5 + weight) * ref + (0.5 - weight) * input_list[i]
            fusion_list.append(i_fusion)
        fusion = torch.div(sum(fusion_list), length)
        return fusion


class BurstSRWeightedFrame(nn.Module):

    def __init__(self, scala=4):
        super(BurstSRWeightedFrame, self).__init__()
        self.srconv = SRConvlayer()
        self.upsample = SRUpsampleBlock(scala=scala)
        self.renew = _FrameWeighted()

    def init_zoo(self, srconv='', upsample='', renew=''):
        if srconv != '':
            print('Loading srconv: %s' % srconv)
            self.srconv.load_state_dict(torch.load(srconv))
        if upsample != '':
            print('Loading upsample: %s' % upsample)
            self.upsample.load_state_dict(torch.load(upsample))
        self.renew.init_zoo(renew=renew)

    def forward(self, frame_list):
        fusion = self.renew(frame_list)
        memory = self.srconv(fusion)
        sr = self.upsample(memory)
        return sr, fusion


class BurstSRWeightedFrameAvgBase(nn.Module):

    def __init__(self, scala=4):
        super(BurstSRWeightedFrameAvgBase, self).__init__()
        self.srconv = SRConvlayer()
        self.upsample = SRUpsampleBlock(scala=scala)
        self.renew = _FrameWeightedAvg()

    def init_zoo(self, srconv='', upsample='', renew=''):
        if srconv != '':
            print('Loading srconv: %s' % srconv)
            self.srconv.load_state_dict(torch.load(srconv))
        if upsample != '':
            print('Loading upsample: %s' % upsample)
            self.upsample.load_state_dict(torch.load(upsample))
        self.renew.init_zoo(renew=renew)

    def forward(self, frame_list):
        fusion = self.renew(frame_list)
        memory = self.srconv(fusion)
        sr = self.upsample(memory)
        return sr, fusion


class BurstImageSRModel(object):
    def __init__(self, logger=None, scala=4, model=BurstSRWeightedSum):
        self.logger = logger
        self.scala = scala
        self.model = model(scala=scala)

    def cuda(self):
        self.model.cuda()

    def name(self):
        return 'BurstImageSRModel'

    def init_zoo(self, srconv='', upsample='', renew='', flow=''):
        self.model.init_zoo(srconv=srconv, upsample=upsample, renew=renew, flow=flow)

    def initialize(self, lr):
        self.lr = lr
        self.optim = torch.optim.Adam(itertools.chain(self.model.srconv.parameters(), self.model.renew.parameters(), self.model.upsample.parameters()), lr=lr)
        self.sr_criterion = nn.L1Loss()
        self._init_loss()

    def initialize_pretrain(self):
        self.optim = torch.optim.Adam(self.model.renew.parameters(), lr=self.lr)

    def _init_loss(self):
        self.loss = {'sr': []}

    def set_input(self, input, gt=None):
        self.input = input
        if gt is not None:
            self.gt = gt
        self.T = len(input)
        self.tensor = torch.cuda.FloatTensor if isinstance(self.gt, torch.cuda.FloatTensor) else torch.Tensor

    def train(self):
        self._init_loss()
        gt_memory = None
        for i in range(self.T):
            if i == 0:
                sr, memory = self.model(self.input[0])
                sr_loss = self.sr_criterion(sr, self.gt)
                gt_memory = memory.data
                self.loss['sr'].append(sr_loss.data[0])
            else:
                old_memory = Variable(gt_memory).cuda()
                sr, memory = self.model(self.input[i], self.input[0], old_memory)
                sr_loss = self.sr_criterion(sr, self.gt)
                loss = sr_loss
                gt_memory = memory.data
                self.loss['sr'].append(sr_loss.data[0])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def test(self, srlist=False):
        sr_results = list()
        gt_memory = None
        print('Processing %d images:' % self.T)
        for i in range(self.T):
            if i == 0:
                sr, memory = self.model(self.input[0])
                gt_memory = memory.data
                sr_results.append(sr)
            else:
                old_memory = Variable(gt_memory)
                sr, memory = self.model(self.input[i], self.input[0], old_memory)
                gt_memory = memory.data
                if srlist:
                    sr_results.append(sr)
                else:
                    sr_results[0] = sr
            print('%d-' % (i+1))
        return sr_results if srlist else sr_results[0]

    def make_input(self, path, frame, channels=1):
        images = _image_file(path)
        self.input = [None] * frame
        self.T = frame
        for i, pic_path in enumerate(images):
            if i >= frame:
                break
            self.input[i] = PIL2VAR(Image.open(pic_path).convert('YCbCr'), norm_function=_sigmoid_to_tanh, volatile=True)[:, :channels, :128, :128]

    def get_current_input(self):
        return self.input

    def _assert_training(self):
        assert self.logger is not None, 'Training'

    def save_model(self, epoch):
        self._assert_training()
        self.save_network(self.model.renew, 'renew', epoch)
        self.save_network(self.model.srconv, 'srconv', epoch)
        self.save_network(self.model.upsample, 'upsample', epoch)

    def load_model(self, epoch):
        self._assert_training()
        self.load_network(self.model.renew, 'renew', epoch)
        self.load_network(self.model.srconv, 'srconv', epoch)
        self.load_network(self.model.upsample, 'upsample', epoch)

    def save_filename(self, network_label, epoch_label):
        self._assert_training()
        if isinstance(epoch_label, int):
            epoch_label = str(epoch_label)
        return '%s_net_%s.pth' % (epoch_label, network_label)

    def save_network(self, network, network_label, epoch_label):
        self._assert_training()
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    def load_network(self, network, network_label, epoch_label):
        self._assert_training()
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        network.load_state_dict(torch.load(save_path))

    def get_current_errors(self):
        return self.loss


class BurstWeightedFrame(object):
    def __init__(self, logger=None, scala=4, model=BurstSRWeightedFrame):
        self.logger = logger
        self.scala = scala
        self.model = model(scala=scala)

    def name(self):
        return 'BurstWeightedFrame'

    def cuda(self):
        self.model.cuda()

    def init_zoo(self, srconv='', upsample='', renew=''):
        self.model.init_zoo(srconv=srconv, upsample=upsample, renew=renew)

    def initialize(self, lr):
        self.lr = lr
        self.optim = torch.optim.Adam(itertools.chain(self.model.srconv.parameters(), self.model.renew.parameters(), self.model.upsample.parameters()), lr=lr)
        self.sr_criterion = nn.L1Loss()
        self._init_loss()

    def initialize_pretrain(self):
        self.optim = torch.optim.Adam(self.model.renew.parameters(), lr=self.lr)

    def _init_loss(self):
        self.loss = {'sr': []}

    def set_input(self, input, gt=None):
        self.input = input
        if gt is not None:
            self.gt = gt
        self.T = len(input)
        self.tensor = torch.cuda.FloatTensor if isinstance(self.gt, torch.cuda.FloatTensor) else torch.Tensor

    def train(self):
        self._init_loss()
        sr, _ = self.model(self.input)
        loss = self.sr_criterion(sr, self.gt)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss['sr'].append(loss.data[0])

    def test(self):
        return self.model(self.input)[0]

    def make_input(self, path, frame, channels=1):
        images = _image_file(path)
        self.input = [None] * frame
        self.T = frame
        for i, pic_path in enumerate(images):
            if i >= frame:
                break
            self.input[i] = PIL2VAR(Image.open(pic_path).convert('YCbCr'), norm_function=_sigmoid_to_tanh, volatile=True)[:, :channels, :256, :256]

    def get_current_input(self):
        return self.input

    def _assert_training(self):
        assert self.logger is not None, 'Training'

    def save_model(self, epoch):
        self._assert_training()
        self.save_network(self.model.renew, 'renew', epoch)
        self.save_network(self.model.srconv, 'srconv', epoch)
        self.save_network(self.model.upsample, 'upsample', epoch)

    def load_model(self, epoch):
        self._assert_training()
        self.load_network(self.model.renew, 'renew', epoch)
        self.load_network(self.model.srconv, 'srconv', epoch)
        self.load_network(self.model.upsample, 'upsample', epoch)

    def save_filename(self, network_label, epoch_label):
        self._assert_training()
        if isinstance(epoch_label, int):
            epoch_label = str(epoch_label)
        return '%s_net_%s.pth' % (epoch_label, network_label)

    def save_network(self, network, network_label, epoch_label):
        self._assert_training()
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    def load_network(self, network, network_label, epoch_label):
        self._assert_training()
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        network.load_state_dict(torch.load(save_path))

    def get_current_errors(self):
        return self.loss


# class BurstImageSRModel(object):
#     def __init__(self, logger, scala=4):
#         self.logger = logger
#         self.scala = scala
#         self.flowNet = MapsFlow()
#         self.renewNet = RenewNet()
#         self.warp = Warp()
#         self.tvloss = TVLoss()
#         self.srconv = SRConvlayer()
#         self.upsample = SRUpsampleBlock(scala=scala)
#
#     def cuda(self):
#         self.flowNet.cuda()
#         self.renewNet.cuda()
#         self.warp.cuda()
#         self.tvloss.cuda()
#         self.srconv.cuda()
#         self.upsample.cuda()
#         self.sr_criterion.cuda()
#         self.tv_criterion.cuda()
#
#     def name(self):
#         return 'BurstImageSRModel'
#
#     def init_zoo(self, srconv='', upsample='', renew='', flow=''):
#         if srconv != '':
#             self.srconv.load_state_dict(torch.load(srconv))
#         if upsample != '':
#             self.upsample.load_state_dict(torch.load(upsample))
#         if renew != '':
#             self.renewNet.load_state_dict(torch.load(renew))
#         if flow != '':
#             self.flowNet.load_state_dict(torch.load(flow))
#
#     def initialize(self, lr, lambda_decay=0.9, lamnda_warp=0.1, lambda_tv=0.1):
#         self.lr = lr
#         self.optim = torch.optim.Adam(itertools.chain(self.srconv.parameters(), self.flowNet.parameters(), self.renewNet.parameters(), self.upsample.parameters()), lr=lr)
#         self.lambda_warp = lamnda_warp
#         self.lambda_tv = lambda_tv
#         self.lambda_decay = lambda_decay
#         self.sr_criterion = nn.L1Loss()
#         self.warp_criterion = lambda x, y: torch.sum(torch.abs(x - y)) / x.data.nelement()
#         self.tv_criterion = TVLoss()
#         self._init_loss()
#
#     def initialize_pretrain(self):
#         self.optim = torch.optim.Adam(itertools.chain(self.flowNet.parameters(), self.renewNet.parameters()), lr=self.lr)
#
#     def _init_loss(self):
#         self.loss = {'sr': [], 'warp': [], 'tv': []}
#
#     def set_input(self, input, gt):
#         self.input = input
#         self.gt = gt
#         self.T = len(input)
#
#     def backward(self):
#         self._init_loss()
#         self.memory = list()
#         self.loss_ = list()
#         memory = self.srconv(self.input[0])
#         self.memory.append(memory)
#         sr_result = self.upsample(self.memory[-1])
#         sr_loss = self.sr_criterion(sr_result, self.gt) * (self.lambda_decay ** (self.T - 1))
#         self.loss_.append(sr_loss)
#         self.loss['sr'].append(sr_loss.data[0])
#         self.loss['warp'].append(0.)
#         self.loss['tv'].append(0.)
#         for i in range(self.T):
#             new_map = self.srconv(self.input[i])
#             flow = self.flowNet(new_map, self.memory[-1])
#             warped_new_map = self.warp(new_map, flow)
#             warp_step_loss = self.warp_criterion(warped_new_map, self.memory[-1]) * self.lambda_warp
#             tv_step_loss = self.tv_criterion(flow) * self.lambda_tv
#             self.loss['warp'].append(warp_step_loss.data[0])
#             self.loss['tv'].append(tv_step_loss.data[0])
#             self.memory.append(self.memory[-1] + self.renewNet(warped_new_map, self.memory[-1]))
#             sr_result = self.upsample(self.memory[-1])
#             sr_step_loss = self.sr_criterion(sr_result, self.gt)
#             self.loss['sr'].append(sr_step_loss.data[0])
#             step_loss = warp_step_loss + tv_step_loss + sr_step_loss
#             self.loss_.append(step_loss)
#         loss = sum(self.loss_)
#         loss.backward()
#
#     def optimize_parameters(self):
#         self.optim.zero_grad()
#         self.backward()
#         self.optim.step()
#
#     def train_step(self, input, gt):
#         self.set_input(input, gt)
#         for i in range(self.T):
#             self.optimize_parameters()
#         return self.get_current_errors()
#
#     def test(self):
#         sr_results = list()
#         memory = self.srconv(self.input[0])
#         sr_results.append(self.upsample(memory))
#         for i in range(1, self.T):
#             new_map = self.srconv(self.input[i])
#             flow = self.flowNet(new_map, memory)
#             warped_new_map = self.warp(new_map, flow)
#             memory += self.renewNet(warped_new_map, memory)
#             sr_results.append(self.upsample(memory))
#         return sr_results
#
#     def save_model(self, epoch):
#         self.save_network(self.flowNet, 'flow', epoch)
#         self.save_network(self.renewNet, 'renew', epoch)
#         self.save_network(self.srconv, 'srconv', epoch)
#         self.save_network(self.upsample, 'upsample', epoch)
#
#     def load_model(self, epoch):
#         self.load_network(self.flowNet, 'flow', epoch)
#         self.load_network(self.renewNet, 'renew', epoch)
#         self.load_network(self.srconv, 'srconv', epoch)
#         self.load_network(self.upsample, 'upsample', epoch)
#
#     def save_filename(self, network_label, epoch_label):
#         if isinstance(epoch_label, int):
#             epoch_label = str(epoch_label)
#         return '%s_net_%s.pth' % (epoch_label, network_label)
#
#     # helper saving function that can be used by subclasses
#     def save_network(self, network, network_label, epoch_label):
#         save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
#         torch.save(network.cpu().state_dict(), save_path)
#         network.cuda()
#
#     # helper loading function that can be used by subclasses
#     def load_network(self, network, network_label, epoch_label):
#         save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
#         network.load_state_dict(torch.load(save_path))
#
#     def get_current_errors(self):
#         return self.loss

