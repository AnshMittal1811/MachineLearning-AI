"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import torch
from auxiliary.my_utils import yellow_print
from model.model import StyleNet
from model.styleatlasnet import StyleAtlasnet
from model.model_blocks import PointNet
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os

class TrainerModel(object):
    def __init__(self):
        """
        This class creates the architectures and implements all trainer functions related to architecture.
        """
        super(TrainerModel, self).__init__()

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if torch.cuda.is_available():
            self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")

        self.network = StyleNet(self.opt)
        self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

        self.reload_network()

        if self.opt.reload_pointnet_path != "":
            self.perceptual_network = PointNet()
            self.perceptual_network.to(self.opt.device)
            self.perceptual_network = nn.DataParallel(self.perceptual_network, device_ids=self.opt.multi_gpu)
            state_dict = torch.load(self.opt.reload_pointnet_path, map_location='cuda:0')
            self.perceptual_network.module.load_state_dict(state_dict)
            self.perceptual_network.eval()

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.reload_model_path != "":
            # print(self.network.state_dict().keys())
            # print(torch.load(self.opt.reload_model_path).keys())

            from collections import OrderedDict
            state_dict = torch.load(self.opt.reload_model_path, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            state_dict = new_state_dict
            """
            pop_list = []
            for k1, k2 in zip(state_dict.keys(), self.network.module.state_dict().keys()):
                if k1 != k2:
                    print('k1: ', k1, '\n', 'k2: ', k2)
                    pop_list.append((k1, k2))
            for k1, k2 in pop_list:
                state_dict[k2] = self.network.module.state_dict()[k2]
                state_dict.pop(k1)
            """
            self.network.module.load_state_dict(state_dict)
            yellow_print(f"Network weights loaded from  {self.opt.reload_model_path}!")
            """
            self.save_network()
            """
        elif self.opt.reload_decoder_path != "":
            opt = deepcopy(self.opt)
            opt.SVR = False
            network = StyleAtlasnet(opt)
            network = nn.DataParallel(network, device_ids=opt.multi_gpu)
            network.module.load_state_dict(torch.load(opt.reload_decoder_path, map_location='cuda:0'))
            self.network.module.decoder = network.module.decoder
            yellow_print(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path}!")
        else:
            yellow_print("No network weights to reload!")

    def reload_best_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.best_model_path != "" and os.path.exists(self.opt.best_model_path):
            yellow_print(f"Network weights loaded from  {self.opt.best_model_path}!")
            # print(self.network.state_dict().keys())
            # print(torch.load(self.opt.reload_model_path).keys())
            from collections import OrderedDict
            state_dict = torch.load(self.opt.best_model_path, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            state_dict = new_state_dict
            self.network.module.load_state_dict(state_dict)
        else:
            yellow_print(f"Failed to reload network weights from  {self.opt.best_model_path}!")

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.discriminator_parameters = (list(self.network.module.discriminator_encoder.parameters()) +
                                    list(self.network.module.discriminator_mlp.parameters()))
        self.generator_parameters = (list(self.network.module.content_encoder.parameters()) +
                                list(self.network.module.style_encoder.parameters()) +
                                list(self.network.module.decoder.parameters()))
        self.generator_optimizer = optim.Adam([p for p in self.generator_parameters if p.requires_grad],
                                              lr=self.opt.generator_lrate)
        self.discriminator_optimizer = optim.Adam([p for p in self.discriminator_parameters if p.requires_grad],
                                                  lr=self.opt.discriminator_lrate)

        if self.opt.reload_generator_optimizer_path != "":
            try:
                self.generator_optimizer.load_state_dict(torch.load(self.opt.reload_generator_optimizer_path,
                                                                    map_location='cuda:0'))
                yellow_print(f"Network weights loaded from  {self.opt.reload_generator_optimizer_path}!")
            except:
                yellow_print(f"Failed to reload optimizer {self.opt.reload_generator_optimizer_path}")
        if self.opt.reload_discriminator_optimizer_path != "":
            try:
                self.discriminator_optimizer.load_state_dict(torch.load(self.opt.reload_discriminator_optimizer_path,
                                                                    map_location='cuda:0'))
                yellow_print(f"Network weights loaded from  {self.opt.reload_discriminator_optimizer_path}!")
            except:
                yellow_print(f"Failed to reload optimizer {self.opt.reload_discriminator_optimizer_path}")


        # Set policy for warm-up if you use multiple GPUs
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()
