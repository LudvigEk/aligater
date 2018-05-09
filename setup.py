#!/usr/bin/env python3
#Start installation
from setuptools import setup, Extension, find_packages

class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

def extensions():
    from Cython.Build import cythonize
    import numpy as np
    sourcefiles = ['aligater/AGCython.pyx', 'aligater/AGc.c']
    ext =Extension("aligater.AGCython", sourcefiles, include_dirs=[np.get_include()])
    ext = Extension("aligater.AGCython", sourcefiles, include_dirs=[np.get_include()])
    return cythonize([ext])


configuration = {
      'name': 'aligater',
      'version': '0.1.0',
      'description': 'Semi automatic gating toolkit',
      'url': 'http://github.com/LudvigEk/aligater',
      'author': 'Ludvig Ekdahl',
      'author_email': 'med-lue@med.lu.se',
      'license': 'MIT',
      'packages': find_packages('aligater'),
      'setup_requires': ['numpy','Cython'],
      'install_requires': ['numpy', 'scipy', 'sklearn','pandas', 'matplotlib','Cython','Jupyter'],
      'ext_modules': lazy_cythonize(extensions),
      'data_files': [('.', ['pyAliGater_testing.ipynb'])],
      'include_package_data': True,
      'Platform': 'Built for ubuntu 16.04+'
}

setup(**configuration)


