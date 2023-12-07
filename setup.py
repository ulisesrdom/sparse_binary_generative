from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


gcc_flags = [] #['-shared', '-O2', '-fno-strict-aliasing']

ext2 = Extension('model_functions',
                sources=['c/c_model_functions.c', 'model_functions.pyx'],
                language='c',
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                #extra_compile_args=['/openmp'],
                #extra_link_args=['/openmp'],
                include_dirs = get_numpy_include_dirs())

ext2.cython_directives = {'language_level': "3"} 

setup(name='FUNCTION MODULES',
      ext_modules = [ext2],
      cmdclass = {'build_ext': build_ext},
      # since the package has c code, the egg cannot be zipped
      zip_safe=False)

