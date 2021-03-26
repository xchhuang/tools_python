from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension


include_dirs = os.path.dirname(os.path.abspath(__file__))

source_file = glob.glob(os.path.join('cpp', '*.cpp'))

setup(
    name='test_cpp',
    ext_modules=[CppExtension('test_cpp', sources=source_file, include_dirs=[include_dirs])],
    cmdclass={
        'build_ext': BuildExtension
    }
)
