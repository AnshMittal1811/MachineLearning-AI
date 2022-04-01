import importlib


def get_model(name):
    module = importlib.import_module("core.models." + name)
    return module.Model
