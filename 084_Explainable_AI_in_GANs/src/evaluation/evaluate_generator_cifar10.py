"""
This file evaluates the generators trained on CIFAR10. Unfortunately, the evaluation code is not ours and so
is not included here. In order to evaluate, use the generated folder of .jpgs at https://github.com/bioinf-jku/TTUR.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from utils.vector_utils import noise_cifar
import torch
import argparse
import numpy as np
from models.generators import GeneratorNetCifar10
import os
from PIL import Image


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/CIFAR100Normal/generator.pt", help="path of the file")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=2048,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics_cifar(path=args.file, numberOfSamples=args.number_of_samples)


def calculate_metrics_cifar(path, numberOfSamples=2048):
    """
    This function is supposed to calculate metrics for cifar.
    :param path: path of the generator model
    :type path: str
    :param numberOfSamples: number of samples to generate
    :type numberOfSamples: int
    :return: None
    :rtype: None
    """
    folder = f'{os.getcwd()}/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    generate_samples_cifar(number=numberOfSamples, path_model=path, path_output=folder)
    return


def generate_samples_cifar(number, path_model, path_output):
    """
    This function generates samples for the CIFAR GAN and saves them as jpg
    :param number: number of samples to generate
    :type number: int
    :param path_model: path where the CIFAR generator is saved
    :type path_model: str
    :param path_output: path of the folder to save the images
    :type path_output: str
    :return: None
    :rtype: None
    """
    generator = GeneratorNetCifar10()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    for i in range(number):
        sample = generator(noise_cifar(1, False)).detach().squeeze(0).numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample/2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{path_output}/{i}.jpg')
    return


if __name__ == "__main__":
    main()
