from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extension = CUDAExtension('mnh_cuda',
    ['utils.cu', 'pybind.cu', 'mlp_eval.cu', 'generate_inputs.cu', 'reorder.cu', 'integrate.cu', 'prebake.cu'],
    extra_compile_args = {'cxx': [], 'nvcc': ['-Xptxas', '-v,-warn-lmem-usage']}
)

setup(
    name='mnh_cuda',
    ext_modules=[extension],
    cmdclass={
        'build_ext': BuildExtension
    }
)
