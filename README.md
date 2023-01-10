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


After downloading, install AliGater by running in the aligater root directory:

	pip3 install -e .

Contribute & Support
---------------------

- Issue Tracker: github.com/LudvigEk/Aligater/issues
- Source Code: github.com/LudvigEk/Aligater

Contact
-------

Lead dev; ludvig.ekdahl@med.lu.se

Citation
--------
Paper in prep;
AliGater: framework for computer-assisted analysis of large-scale flow-cytometry data. Ekdahl et. al.

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
