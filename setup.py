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
      install_requires=['numpy','pandas', 'seaborn','cython'],
      zip_safe=False)

sourcefiles = ['aligater/AGCython.pyx', 'aligater/AGc.c']

extensions = [Extension("aligater.AGCython", sourcefiles)]

cythonsetup(
    ext_modules = cythonize(extensions)
)
