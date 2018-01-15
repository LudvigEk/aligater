from setuptools import setup

setup(name='aligater',
      version='0.1',
      description='Semi automatic gating toolkit',
      url='http://github.com/LudvigEk/aligater',
      author='Ludvig Ekdahl',
      author_email='med-lue@med.lu.se',
      license='MIT',
      packages=['aligater'],
      install_requires=['numpy','pandas', 'seaborn'],
      zip_safe=False)
