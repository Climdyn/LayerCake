# LayerCake

> “Welcome to the layer cake son.”
>
> -- *Michael Gambon - Layer Cake (2004)*

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

(c) 2025-2026 Jonathan Demaeyer

See [LICENSE.txt](./LICENSE.txt) for license information.

## Installation

### With pip

Not yet available.

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

## Examples

A few examples are available in the [examples](./examples) folder. More examples will be provided as the code is 
developed.

## Contributing

LayerCake is in betaa development phase, bug reports and tests of the features are welcome.
Please simply raise an issue on [Github](https://github.com/Climdyn/LayerCake/issues).
