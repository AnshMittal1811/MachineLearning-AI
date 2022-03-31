"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='shaper-comp_ext',
    ext_modules=[
        CUDAExtension(
            name='fps_cuda',
            sources=[
                'cuda/fps.cpp',
                'cuda/fps_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='nn_distance_cuda',
            sources=[
                'cuda/nn_distance.cpp',
                'cuda/nn_distance_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='knn_distance_cuda',
            sources=[
                'cuda/knn_distance.cpp',
                'cuda/knn_distance_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='group_points_cuda',
            sources=[
                'cuda/group_points.cpp',
                'cuda/group_points_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
         CUDAExtension(
             name='chamfer_cuda',
             sources=[
                 'cuda/chamfer.cpp',
                 'cuda/chamfer_kernel.cu',
             ],
             extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
         ),
        # CUDAExtension(
        #     name='ball_query_cuda',
        #     sources=[
        #         'cuda/ball_query.cpp',
        #         'cuda/ball_query_kernel.cu',
        #     ],
        #     extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        # ),
        # CUDAExtension(
        #     name='emd_cuda',
        #     sources=[
        #         'cuda/emd.cpp',
        #         'cuda/emd_kernel.cu',
        #     ],
        #     extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        # ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
