# LayerCake

> “Welcome to the layer cake son.”
>
> -- <cite>Michael Gambon - Layer Cake (2004)</cite>

## General Information

LayerCake is framework to design systems of partial differential equations (PDEs), 
and convert them to ordinary differential equations (ODEs) via a Galerkin-type expansion.

LayerCake allows you to construct systems of PDEs, and to specify coordinate systems and basis functions to compute
the ODEs.

## About

(c) 2025 Jonathan Demaeyer

See [LICENSE.txt](./LICENSE.txt) for license information.

## Installation

### With pip

Not yet avalialble.

### With Anaconda

The second easiest way to install and run LayerCake is to use an appropriate environment 
created through [Anaconda](https://www.anaconda.com/).

First install Anaconda and clone the repository:

    git clone https://github.com/jodemaey/LayerCake.git

Then install and activate the Python3 Anaconda environment:

    conda env create -f environment.yml
    conda activate layercake

You can then perform a test by running the script

    python examples/atmospheric/barotropic_one_layer.py
    
to see if everything runs smoothly (this should take less than 5 minutes to run).

## Examples

A few examples are available in the [examples](./examples) folder. More examples will be provided as the code is 
developed.

## Forthcoming development

LayerCake is in alpha development phase, with many new functionalities planned over the next few months.
Stay tuned !
