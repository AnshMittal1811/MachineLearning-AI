# Modified here to use c complier for cuda 11
# https://github.com/autonomousvision/occupancy_networks/issues/96
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
import numpy

pykdtree = Extension(
    'core.models.utils.occnet_utils.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'core/models/utils/occnet_utils/utils/libkdtree/pykdtree/kdtree.c',
        'core/models/utils/occnet_utils/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

ext_modules = [
    pykdtree,
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmd_class={
        'build_ext': build_ext
    }
)