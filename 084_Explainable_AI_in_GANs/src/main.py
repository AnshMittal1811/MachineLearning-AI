"""
This file is the main function that parses user argument and runs experiments.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import argparse
from experiment_enums import experimentsAll


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run xAI-GAN experiment using the provided experiment enums.")
    args = parser.parse_args()
    experiment_setup(args)


def experiment_setup(args: argparse.Namespace) -> None:
    """
    This function sets up the experiment and runs it for both regular and logic GANs
    :param args: dictionary arguments from user
    :return: None
    """
    experiments = experimentsAll
    for experiment in experiments:
        experiment.run(logging_frequency=1)


if __name__ == "__main__":
    main()
