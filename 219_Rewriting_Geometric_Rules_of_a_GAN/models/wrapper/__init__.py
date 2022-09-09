import importlib

from .base_wrapper import BaseWrapper
from lib.dissect.param_tool import merge_state_dict


def find_wrapper_using_name(wrapper_name):
    # Given the option [wrappername],
    # the file "wrappers/wrappername_wrapper.py"
    # will be imported.
    wrapper_filename = "models.wrapper." + wrapper_name + "_wrapper"
    wrapperlib = importlib.import_module(wrapper_filename)

    # In the file, the class called WrapperNameWrapper() will
    # be instantiated. It has to be a subclass of BaseWrapper,
    # and it is case-insensitive.
    wrapper = None
    target_wrapper_name = wrapper_name.replace('_', '') + 'wrapper'
    for name, cls in wrapperlib.__dict__.items():
        if name.lower() == target_wrapper_name.lower() \
                and issubclass(cls, BaseWrapper):
            wrapper = cls

    if wrapper is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (wrapper_filename, target_wrapper_name))
        exit(0)

    return wrapper


def create_wrapper(wrapper_name, module, module_name, **kwargs):
    wrapper = find_wrapper_using_name(wrapper_name)
    instance = wrapper(module, module_name, **kwargs)
    return instance


def create_wrappers(wrapper_name, module_dict, **kwargs):
    wrappers = []
    for name, module in module_dict.items():
        wrapper = find_wrapper_using_name(wrapper_name)
        instance = wrapper(module, name, **kwargs)
        wrappers.append(instance)
    return wrappers


def weight_reg_loss_from_wrappers(wrappers):
    return sum([w.get_weight_reg_loss() for w in wrappers])


def update_params_from_wrappers(wrappers):
    return merge_state_dict(*[w.get_update_params() for w in wrappers])


def save_params_from_wrappers(wrappers):
    return merge_state_dict(*[w.get_save_params() for w in wrappers])
