import torch
from distutils.version import LooseVersion

if LooseVersion(torch.__version__) >= LooseVersion('1.11.0'):
    # New conv refactoring started at version 1.11, it seems.
    from .conv2d_gradfix_111andon import conv2d, conv_transpose2d
else:
    from .conv2d_gradfix_pre111 import conv2d, conv_transpose2d