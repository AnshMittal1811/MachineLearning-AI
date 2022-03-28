"""
This file evaluates the generators trained on MNIST and Fashion MNIST. Unfortunately the evaluation code is not ours and
so is not included here. In order to evaluate, use the saved numpy files at https://github.com/mseitzer/pytorch-fid

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from utils.vector_utils import noise
import torch
import argparse
import numpy as np
from get_data import get_loader
from models.generators import GeneratorNet


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/MNIST10Normal/generator.pt", help="path of the file")
    parser.add_argument("-t", "--type", default="fmnist", help="type of the dataset")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=10000,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics(path=args.file, numberOfSamples=args.number_of_samples, datasetType=args.type)


def calculate_metrics(path, numberOfSamples=10000, datasetType="mnist"):
    """
    This function calculates fid score for mnist and fashion mnist models.
    :param path: the path of the saved generator model
    :type path: str
    :param numberOfSamples: the number of samples to generate
    :type numberOfSamples: int
    :param datasetType: the type of dataset (mnist or fmnist)
    :type datasetType: str
    :return: None
    :rtype: None
    """
    # get real data
    path_real = "./real.npy"
    generate_real_data(number=numberOfSamples, path=path_real, datasetType=datasetType)

    # get generated data
    path_generated = "./generated.npy"
    generate_samples(number=numberOfSamples, path_model=path, path_output=path_generated)
    return


def generate_real_data(number: int, path: str, datasetType="mnist") -> None:
    """
    This function saves a batch of data from either MNIST or Fashion MNIST models
    :param number:
    :type number:
    :param path:
    :type path:
    :param datasetType:
    :type datasetType:
    :return:
    :rtype:
    """
    # get background data and save
    loader = get_loader(number, 1, datasetType)
    batch = next(iter(loader))[0].detach()
    batch = batch.view(number, 1, 32, 32)
    np.save(path, batch)


def generate_samples(number, path_model, path_output):
    """
    This function generates samples for MNIST and Fashion MNIST generators
    :param number: the number of samples to generate
    :type number: int
    :param path_model: the path of the saved generator model
    :type path_model: str
    :param path_output: path of the folder to save generated samples
    :type path_output: str
    :return: None
    :rtype: None
    """
    generator = GeneratorNet()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    samples = generator(noise(number, False)).detach()
    samples = samples.view(number, 1, 32, 32)
    np.save(path_output, samples)


if __name__ == "__main__":
    main()
