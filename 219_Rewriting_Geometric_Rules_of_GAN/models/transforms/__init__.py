import importlib
import torch


def find_transform_using_name(transform_name):
    # Given the option --transform [transformname],
    # the file "transforms/transformname_transform.py"
    # will be imported.
    transform_filename = "models.transforms." + transform_name + "_transform"
    transformlib = importlib.import_module(transform_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    transform = None
    target_transform_name = transform_name.replace('_', '') + 'transform'
    for name, cls in transformlib.__dict__.items():
        if name.lower() == target_transform_name.lower() \
                and issubclass(cls, torch.nn.Module):
            transform = cls

    if transform is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (transform_filename, target_transform_name))
        exit(0)

    return transform


def get_option_setter(transform_name):
    transform_class = find_transform_using_name(transform_name)
    return transform_class.modify_commandline_options


def create_transform(opt):
    transform = find_transform_using_name(opt.transform)
    instance = transform(opt)
    print("transform [%s] was created" % (type(instance).__name__))

    return instance
