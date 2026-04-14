User guide
==========

This guide explains how the LayerCake framework can be used to transform a set of two-dimensional partial differential

.. math::

    \partial_t \mathcal{F}^{\mathrm LHS}_i (\psi_1, \ldots, \psi_N) = \mathcal{F}^{\mathrm{RHS}}_i (\psi_1, \ldots, \psi_N) \qquad , \quad i = 1,\ldots,N
equations (PDEs) defined on a particular domain into a system of ordinary differential equations (ODEs)
with an automated `Galerkin method`_. This method projects all the fields :math:`\psi_j` on given function basis :math:`\phi_{j,k}`:

.. math::

    \psi_j = \sum_{k=1}^{n_j} \psi_{j, k} \,\, \phi_{j,k}

and the resulting discrete representation of the spatially continuous model defined by the PDEs is sometimes called its
(truncated) representation in the spectral domain.

The full system of PDEs is called the `cake`, and the system of equations can be divide
into different subsets called `layers`.


1. Rationale behind LayerCake
-----------------------------

The obtained ODEs

.. math:: \dot{\boldsymbol{x}} = \boldsymbol{f}(t, \boldsymbol{x})

are considered to be the model that the user is looking after, and :math:`\boldsymbol{x}` is a state vector consisting
of the stacked spectral coefficients :math:`\boldsymbol{x} = (\psi_{1,1},\ldots,\psi_{1,n_1}, \ldots, \psi_{N,1},\ldots,\psi_{N,n_N})`.
The purpose of LayerCake is then to provide a Python callable or a set of strings representing
the model's tendencies :math:`\boldsymbol{f}` (and also its Jacobian matrix :math:`\boldsymbol{D f}`) that can be
used for example to

* integrate the model over time
* perform sensitivity analysis
* perform bifurcation analysis

To obtain these ODEs, the user must first specifies the PDEs system, its parameters, and its domain
(represented by a coordinate system and a set of basis functions).
In particular, the specification of the PDEs system is done by constructing each PDE one by one, adding terms to
a couple of lists representing the LHS and RHS part of the equation.
These terms are provided as :class:`~layercake.arithmetic.terms.base.ArithmeticTerms` objects representing various specific
functionals of the both fields :math:`\psi_j` of the equations and the model's spatial parameter fields.

2. Starting a new model
-----------------------

In general, one will start a new Python script to define a new model.
Here, we are going to detail a simple model available in the `examples/atmospheric <../../../examples/atmospheric/>`_ folder,
i.e. a two-layer quasi-geostrophic model on a `beta-plane`_ with an orography described in Vallis book :cite:`user-V2017` :

.. math::

    \partial_t \left( \nabla^2 \psi_1 + \alpha'_1 \psi_2 + (\alpha_1 - \alpha'_1) \psi_1 \right) & = & - J \left(\psi_1, \nabla^2 \psi_1\right) - J \left(\psi_1, \alpha'_1  \psi_2\right) - \beta \, \partial_x \psi_1 \\
    \partial_t \left( \nabla^2 \psi_2 + \alpha'_2 \left(\psi_1 - \psi_2 \right) \right) & = & - J \left(\psi_2, \nabla^2 \psi_2\right) - J \left(\psi_2, \alpha'_2  \psi_1\right) - \beta \, \partial_x \psi_2 - J \left(\psi_2, h\right)

where :math:`\psi_1, \psi_2` are the non-dimensional fields of the problem, and :math:`h` is the non-dimensional `orography`_ of the model.
:math:`\beta` is the non-dimensional `beta coefficient`_ and the :math:`\alpha_i, \alpha'_i` are coefficients of the model.
:math:`x` and :math:`y` are the non-dimensional coordinates on the beta-plane, and :math:`J(S,G) = \partial_x S \partial_y G - \partial_y S \partial_x G` is the Jacobian of the
fields, typically representing the advection of fluid parcels in the model.

As a first step to implement this model in LayerCake, the script starts with the classic import of the needed classes and functions.

2.1 Importing LayerCake classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Importing all the classes needed to specify the PDEs can be done by simply typing:

.. code:: ipython3

    # importing all that is needed to create the cake
    from layercake import *

This will import most of what is needed to build the `cake`, but a more specific import must be done to define the model's domain:

.. code:: ipython3

    # importing specific modules to create the model basis of functions
    from layercake.basis.planar_fourier import contiguous_channel_basis
    from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition

where we have imported a specific basis of functions definition, and the standard inner product definitions for them.

References
----------

.. bibliography:: bib/ref.bib
    :labelprefix: USER-
    :keyprefix: user-


.. _Galerkin method: https://en.wikipedia.org/wiki/Galerkin_method
.. _orography: https://en.wikipedia.org/wiki/Orography
.. _beta coefficient: https://en.wikipedia.org/wiki/Beta_plane
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane