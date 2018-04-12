#!/usr/bin/python3
from setuptools import setup
from distutils.core import setup as cythonsetup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(name='aligater',
      version='0.1',
      description='Semi automatic gating toolkit',
      url='http://github.com/LudvigEk/aligater',
      author='Ludvig Ekdahl',
      author_email='med-lue@med.lu.se',
      license='MIT',
      packages=['aligater'],
      install_requires=['numpy', 'scipy', 'sklearn','pandas', 'matplotlib','Cython','Jupyter'],
      zip_safe=False)

from distutils.core import setup as cythonsetup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

sourcefiles = ['aligater/AGCython.pyx', 'aligater/AGc.c']

extensions = [Extension("aligater.AGCython", sourcefiles, include_dirs=[np.get_include()])]

cythonsetup(
    ext_modules = cythonize(extensions)
)
