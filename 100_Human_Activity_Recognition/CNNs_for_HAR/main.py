import argparse
import sys
import os

import torch
from torch import optim
from torch import nn

from utils.datasets import get_dataloaders, get_data_size, get_num_classes, get_class_labels, DATASETS
from utils.helpers import get_config_section, FormatterNoDuplicate, set_seed, create_safe_directory
from classifier.cnn import init_specific_model
from classifier.training import Trainer
from classifier.evaluate import Evaluator
from classifier.utils.modelIO import save_model, load_model, load_metadata

from classifier.cnn import MODELS

CONFIG_FILE = "hyperparams.ini"
RES_DIR = "results"

def parse_arguments(args_to_parse):
    """Parse the command line arguments.
        Parameters
        ----------
        args_to_parse: list of str
            Arguments to parse (split on whitespaces).
    """
    description = "PyTorch implementation of CNN's for Human Activity Recognition"
    default_config = get_config_section([CONFIG_FILE], "Preset")
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-s', '--is_standardized', type=bool,
                          default=default_config['is_standardized'],
                          help='Whether to standardize the data.')

    # Model Options
    model = parser.add_argument_group('Model specific options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder to use.')

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-n', '--name', type=str, default=default_config['name'],
                         help="Name of the model for storing and loading purposes.")

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether or not to compute the test losses.`")

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.
        Parameters
        ----------
        args: argparse.Namespace
            Arguments
    """

    set_seed(args.model_type)
    exp_dir = os.path.join(RES_DIR, args.name)

    if not args.is_eval_only:
        # Create directory (if same name exists, archive the old one)
        create_safe_directory(exp_dir)

        # PREPARES DATA
        train_loader = get_dataloaders(args.dataset,
                                       is_train=True,
                                       batch_size=args.batch_size,
                                       is_standardized=args.is_standardized)

        # PREPARES MODEL
        args.data_size = get_data_size(args.dataset)
        args.num_classes = get_num_classes(args.dataset)
        model = init_specific_model(args.model_type, args.data_size, args.num_classes)

        # PREPARES OPTIMIZER AND CRITERION
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # TRAINS
        print('***************************************************')
        print('*                 Training Model                  *')
        print('***************************************************')
        trainer = Trainer(model, optimizer, criterion)
        trainer(train_loader, args.epochs)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if not args.no_test:
        # LOADS MODEL
        model = load_model(exp_dir, is_gpu=False)
        metadata = load_metadata(exp_dir)
        args.num_classes = get_num_classes(args.dataset)

        # GET TRAIN AND TEST DATA
        train_loader = get_dataloaders(metadata["dataset"],
                                       is_train=True,
                                       batch_size=args.batch_size,
                                       is_standardized=args.is_standardized)

        test_loader = get_dataloaders(metadata["dataset"],
                                      is_train=False,
                                      batch_size=args.batch_size,
                                      is_standardized=args.is_standardized)

        # EVALUATE FOR TRAIN AND TEST
        class_labels = get_class_labels(args.dataset)
        evaluate = Evaluator(model, args.num_classes, class_labels)

        print('***************************************************')
        print('*            Evaluating Train Accuracy            *')
        print('***************************************************')
        train_accuracy, class_train_accuracy, confusion = evaluate(train_loader)
        print('Train accuracy of the network on the %i train sequences: %.2f %%' %
              (len(train_loader.dataset), train_accuracy))
        for i in range(args.num_classes):
            print('Accuracy of class %i: %5s : %.2f %%' % (i, class_labels[i], class_train_accuracy[i]))
        print("\n")
        print("Confusion matrix:")
        print("--------------------------------")
        print(confusion)
        print("\n")

        print('***************************************************')
        print('*            Evaluating Test Accuracy             *')
        print('***************************************************')
        test_accuracy, class_test_accuracy, confusion = evaluate(test_loader)
        print('Test accuracy of the network on the %i test sequences: %.2f %%' %
              (len(test_loader.dataset), test_accuracy))
        for i in range(args.num_classes):
            print('Accuracy of class %i: %5s : %.2f %%' % (i, class_labels[i], class_test_accuracy[i]))
        print("\n")
        print("Confusion matrix:")
        print("--------------------------------")
        print(confusion)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)