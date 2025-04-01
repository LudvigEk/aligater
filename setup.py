#!/usr/bin/env python3
# Start installation
from setuptools import setup, Extension, find_packages
from os import environ

if 'CFLAGS' in environ:
    if isinstance(environ['CFLAGS'], tuple):
        # Can this be a tuple?
        raise TypeError
    # Check if already list of flags
    if isinstance(environ['CFLAGS'], list):
        # If that's the case, just append
        environ['CFLAGS'].append("-I. -include glibc_version_fix.h")
    else:
        # If not list, i.e string
        environ['CFLAGS'] = [environ['CFLAGS'], "-I. -include glibc_version_fix.h"]
else:
    # CFLAGS environ variable didn't exist
    environ['CFLAGS'] = "-I. -include glibc_version_fix.h"


class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list(): yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    from Cython.Build import cythonize
    import numpy as np
    sourcefiles = ['aligater/AGCython.pyx']
    # ext =Extension("aligater.AGCython", sourcefiles,libraries=["m"],language='c++', include_dirs=[np.get_include(),"./aligater"])
    ext = Extension("aligater.AGCython", sourcefiles, language='c++', include_dirs=[np.get_include(), "./aligater"])
    sourcefiles1 = ['aligater/AGCythonUtils.pyx']
    # ext1 =Extension("aligater.AGCythonUtils", sourcefiles1,libraries=["m"],language='c++', include_dirs=[np.get_include(),"./aligater"])
    ext1 = Extension("aligater.AGCythonUtils",
                     sourcefiles1,
                     language='c++',
                     include_dirs=[np.get_include(), "./aligater"])
    return cythonize([ext, ext1])


configuration = {
    'name': 'aligater',
    'version': '0.1.0',
    'description': 'Computer-assisted gating toolkit',
    'url': 'http://github.com/LudvigEk/aligater',
    'author': 'Ludvig Ekdahl',
    'author_email': 'ludvig.ekdahl@med.lu.se',
    'license': 'MIT',
    'packages': find_packages('aligater'),
    'setup_requires': ['numpy', 'Cython'],
    'install_requires': ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'Cython', 'h5py', 'ray'],
    'ext_modules': extensions(),
    'package_dir': {'.': 'aligater'},
    'data_files': [('.', [])],
    'include_package_data': True,
    'platforms': 'Built for ubuntu 16.04+'
}

setup(**configuration)
