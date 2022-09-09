import importlib
from trainers.base_trainer import BaseTrainer


def find_trainer_using_name(trainer_name):
    """Import the module "trainers/[trainer_name]_trainer.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseTrainer,
    and it is case-insensitive.
    """
    trainer_filename = "trainers." + trainer_name + "_trainer"
    trainerlib = importlib.import_module(trainer_filename)
    trainer = None
    target_trainer_name = trainer_name.replace('_', '') + 'trainer'
    for name, cls in trainerlib.__dict__.items():
        if name.lower() == target_trainer_name.lower() \
           and issubclass(cls, BaseTrainer):
            trainer = cls

    if trainer is None:
        print("In %s.py, there should be a subclass of BaseTrainer with class name that matches %s in lowercase." % (trainer_filename, target_trainer_name))
        exit(0)

    return trainer


def get_option_setter(trainer_name):
    """Return the static method <modify_commandline_options> of the trainer class."""
    trainer_class = find_trainer_using_name(trainer_name)
    return trainer_class.modify_commandline_options


def create_trainer(opt):
    """Create a trainer given the option.
    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from trainers import create_trainer
        >>> trainer = create_trainer(opt)
    """
    trainer = find_trainer_using_name(opt.trainer)
    instance = trainer(opt)
    return instance
