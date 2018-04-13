#!/usr/bin/env python3
#Check for pre-requisites
import pip3
installed_packages = pip3.get_installed_distributions()
flat_installed_packages = [package.project_name for package in installed_packages]
if 'numpy' not in flat_installed_packages:
	pip3 install --upgrade numpy
if 'Cython' not in flat_installed_packages:
	pip3 install --upgrade Cython
if 'setuptools' not in flat_installed_packages:
	pip3 install --upgrade setuptools

#Start installation
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
sourcefiles = ['aligater/AGCython.pyx', 'aligater/AGc.c']
extensions = [Extension("aligater.AGCython", sourcefiles, include_dirs=[np.get_include()])]

setup(name='aligater',
      version='0.1.0',
      description='Semi automatic gating toolkit',
      url='http://github.com/LudvigEk/aligater',
      author='Ludvig Ekdahl',
      author_email='med-lue@med.lu.se',
      license='MIT',
      packages=['aligater'],
      setup_requires=['numpy','Cython'],
      install_requires=['numpy', 'scipy', 'sklearn','pandas', 'matplotlib','Cython','Jupyter'],
      ext_modules = cythonize(extensions)
)

