import numpy as np


def generate_grid(dim, resolution):
    grid = np.stack(
        np.meshgrid(*([np.arange(resolution)] * dim), indexing="ij"), axis=-1
    )
    grid = (2 * grid + 1) / resolution - 1
    return grid
