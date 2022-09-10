from typing import Tuple

import numpy as np


def adjust_intrinsic(intrinsic: np.array, intrinsic_image_dim: Tuple, image_dim: Tuple) -> np.array:
    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]

    intrinsic_return[0, 0] *= float(width_after) / float(width_before)
    intrinsic_return[1, 1] *= float(height_after) / float(height_before)

    # account for cropping/padding here
    intrinsic_return[0, 2] *= float(width_after - 1) / float(width_before - 1)
    intrinsic_return[1, 2] *= float(height_after - 1) / float(height_before - 1)

    return intrinsic_return
