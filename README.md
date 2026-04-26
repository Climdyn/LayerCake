# LayerCake

> “Welcome to the layer cake son.”
>
> -- *Michael Gambon - Layer Cake (2004)*

[![Documentation Status](https://img.shields.io/badge/docs-passing-green.svg)](https://climdyn.github.io/LayerCake/)

## General Information

LayerCake is a framework to design models based on systems of partial differential equations (PDEs), 
and convert them to ordinary differential equations (ODEs) via Galerkin-type expansions.

LayerCake allows you to construct systems of PDEs, and to specify coordinate systems and basis functions to build
the corresponding ODE systems.
To build these systems, LayerCake relies heavily on the [Sympy](https://www.sympy.org/) symbolic computation framework.
The output of this procedure is either a [Numbaified](https://numba.pydata.org/) Python callable, 
or a list of symbolic ODE tendencies that can be integrated in any of the supported languages 
(Fortran, Julia and Python for the moment). These two kinds of output allow for the study of the computed models with 
the modern tools available in all these languages.

LayerCake has been designed with geophysics in mind, although it may be useful for other applications.

## About

(c) 2025-2026 Jonathan Demaeyer and Oisín Hamilton

See [LICENSE.txt](https://github.com/Climdyn/LayerCake/blob/main/LICENSE.txt) for license information.

## Installation

### With pip

The easiest way to install and run LayerCake is to use [pip](https://pypi.org/).
Type in a terminal

    pip install layercake-model

and you are set!

Additionally, you can clone the repository

    git clone https://github.com/Climdyn/LayerCake.git

and perform a test by running the script

    python examples/atmospheric/barotropic_one_layer.py

to see if everything runs smoothly (this should take less than 5 minutes).

### With Anaconda

The second-easiest way to install and run LayerCake is to use an appropriate environment 
created through [Anaconda](https://www.anaconda.com/).

First install Anaconda and clone the repository:

    git clone https://github.com/jodemaey/LayerCake.git

Then install and activate the Python3 Anaconda environment:

    conda env create -f environment.yml
    conda activate layercake

You can then perform a test by running the script

    python examples/atmospheric/barotropic_one_layer.py
    
to see if everything runs smoothly (this should take less than 5 minutes to run).

## Documentation

To build the documentation, please run (with the conda environment activated):

    cd documentation
    make html


You may need to install [make](https://www.gnu.org/software/make/) if it is not already present on your system.
Once built, the documentation is available [here](./documentation/build/html/index.html).

The documentation is also available online at https://climdyn.github.io/LayerCake/. In particular, 
please consider reading the [User guide](https://climdyn.github.io/LayerCake/files/user_guide.html#).

## Examples

A few examples are available in the [examples](./examples) folder. More examples will be provided as the code is 
further developed.

## Dependencies

LayerCake needs mainly:

* [Numpy](https://numpy.org/) for numeric support
* [sparse](https://sparse.pydata.org/) for sparse multidimensional arrays support
* [Numba](https://numba.pydata.org/) for code acceleration
* [Sympy](https://www.sympy.org/) for symbolic manipulation of inner products

Check the YAML file [environment.yml](https://raw.githubusercontent.com/Climdyn/LayerCake/main/environment.yml) for the dependencies.

## Contributing

LayerCake is in beta development phase, bug reports and tests of the features are welcome.
Please simply raise an issue on [Github](https://github.com/Climdyn/LayerCake/issues).

If you want to contribute actively to the development, please contact the main authors.
In addition, if you have made changes that you think will be useful to others, please feel free to suggest these as a pull request 
on the [LayerCake Github repository](https://github.com/Climdyn/LayerCake/pulls).
