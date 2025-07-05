from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="fast_activations",
    ext_modules=[
        CUDAExtension(
            'fast_activations._C',
            sources=[
                'fast_activations/bindings.cpp',
                'fast_activations/cuda/relu.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)