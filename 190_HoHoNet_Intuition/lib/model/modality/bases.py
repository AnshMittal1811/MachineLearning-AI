import torch
import numpy as np


def dct(n_components, output_height):
    basis = (torch.arange(output_height)[None].float() + 0.5) / output_height * np.pi
    basis = torch.arange(0, n_components)[:,None].float() * basis
    basis = torch.cos(basis)
    return basis


def linear(*args, **kwargs):
    return None
