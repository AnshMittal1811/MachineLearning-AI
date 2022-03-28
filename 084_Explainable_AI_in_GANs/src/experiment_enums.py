"""
This file contains the enums with the details to run experiments. Some simple examples are given below. In order to
create your own experiment, simply fill all the keys.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from enum import Enum
from models.generators import GeneratorNet, GeneratorNetCifar10
from models.discriminators import DiscriminatorNet, DiscriminatorNetCifar10
from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    FMNIST35Normal = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST100Saliency = {
        "explainable": True,
        "explanationType": "saliency",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST100Saliency = {
        "explainable": True,
        "explanationType": "saliency",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST100Shap = {
        "explainable": True,
        "explanationType": "shap",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST100Shap = {
        "explainable": True,
        "explanationType": "shap",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST100Lime = {
        "explainable": True,
        "explanationType": "lime",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST100Lime = {
        "explainable": True,
        "explanationType": "lime",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    # DemoCIFAR = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.5,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }
    #
    # DemoMNIST = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }
    #
    # DemoFMNIST = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }

    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
