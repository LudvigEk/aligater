#!/usr/bin/python3
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['aligater/AGCython.pyx', 'aligater/AGc.c']
extensions = [Extension("aligater.AGCython", sourcefiles, include_dirs=[np.get_include()])]

setup(name='aligater',
      version='0.1',
      description='Semi automatic gating toolkit',
      url='http://github.com/LudvigEk/aligater',
      author='Ludvig Ekdahl',
      author_email='med-lue@med.lu.se',
      license='MIT',
      packages=['aligater'],
      install_requires=['numpy', 'scipy', 'sklearn','pandas', 'matplotlib','Cython','Jupyter'],
      ext_modules = cythonize(extensions),
      zip_safe=False)

)
