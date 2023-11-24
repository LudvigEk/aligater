AliGater
========

AliGater is intended as a rapid application development environment for high-throughput gating using pattern recognition functions and libraries. It provides a framework with basic gating functionality and then lets you build your own mathematical and pattern recognition functions in your strategies.

Typically you can explore and gate your data in a notebook environment then run thousands of gates using the same strategy. See sample notebooks (to be made) for examples.

Jupyter templates for evaluating QC objects (downsampled image views) using clustering will be made available.

Check out the documentation at
https://aligater.readthedocs.io/en/latest/

Features
--------

    * Several pre-built functions to make gating easier, from simple thresholding and fixed quadgates to 1-2d mixed gaussian modelling, dijkstras shortest path implementations and principal components.

    * Support for crunching through folder hierarchies with sample files and linking folder names to your experiment.

    * Straight-forward requirements to build your own pattern-recognition methods into the workflow.

    * Easily integrates with methods of libraries like scikit-learn & scipy.


Installation
------------

Download AliGater by running below:

	git clone https://github.com/LudvigEk/aligater
    cd aligater

After downloading, if you do not have cython installed in your environment it's recommended to pre-install it.

	pip install cython

You should then be able to install AliGater by running:

    pip install -e .

Getting AliGater up and running in a jupyter notebook environment
------------
If you installed AliGater in a local environment such as conda, a few extra steps might be necessary to set up 
a jupyter kernel with the environment containing aligater. Below commands should work for a conda environment.

    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=aligater

Where --name=aligater is the name of the local conda environment where aligater was installed.

Contribute & Support
---------------------

- Issue Tracker: github.com/LudvigEk/Aligater/issues
- Source Code: github.com/LudvigEk/Aligater

Contact
-------

Lead dev; ludvig.ekdahl@med.lu.se

Citation
--------
AliGater: a framework for the development of bioinformatic pipelines for large-scale, high-dimensional cytometry data

Bioinformatics Advances, Volume 3, Issue 1, 2023, vbad103, https://doi.org/10.1093/bioadv/vbad103

License
-------

MIT

Copyright (c) 2023 Ludvig Ekdahl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
