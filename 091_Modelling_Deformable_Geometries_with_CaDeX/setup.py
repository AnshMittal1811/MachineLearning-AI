try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()
# print(numpy_include_dir)

# Extensions
# pykdtree (kd tree)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'core.models.utils.occnet_utils.utils.libmcubes.mcubes',
    sources=[
        'core/models/utils/occnet_utils/utils/libmcubes/mcubes.pyx',
        'core/models/utils/occnet_utils/utils/libmcubes/pywrapper.cpp',
        'core/models/utils/occnet_utils/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'core.models.utils.occnet_utils.utils.libmesh.triangle_hash',
    sources=[
        'core/models/utils/occnet_utils/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'core.models.utils.occnet_utils.utils.libmise.mise',
    sources=[
        'core/models/utils/occnet_utils/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'core.models.utils.occnet_utils.utils.libsimplify.simplify_mesh',
    sources=[
        'core/models/utils/occnet_utils/utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'core.models.utils.occnet_utils.utils.libvoxelize.voxelize',
    sources=[
        'core/models/utils/occnet_utils/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


# Gather all extension modules
ext_modules = [
    # pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=[numpy_include_dir],
)