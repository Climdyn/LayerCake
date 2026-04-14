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
(truncated) representation in the spectral domain [#basis]_. The spectral coefficients :math:`\psi_{j,k}` are determined by taking a specific inner product
:math:`\langle \, , \rangle` between the fields :math:`\psi_j` and the basis of functions :math:`\phi_{j,k}`:

.. math::

    \psi_{j,k} = \left\langle \phi_{j,k} , \psi_j \right\rangle

One can also apply the inner product to the whole equations above, de facto projecting them onto subspaces spanned by the basis of functions.
This procedure results into sets of ODEs for the coefficients :math:`\psi_{j,k}`.
The goal of LayerCake is to produce numerical or symbolic representation of the tendencies of these sets of ODEs.

.. note::

    Some terminology: In LayerCake, the full system of PDEs is called the `cake`, and the system of equations can be divided
    into different subsets called `layers`.


1. Rationale behind LayerCake
-----------------------------

The ODEs resulting from the Galerkin procedure

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
fields, typically representing the advection of physical quantities (like the vorticity) in the model.
This model will be defined on a plane with as boundary conditions walls in the :math:`y` direction at the border, and periodicity in the :math:`x` direction.
This is imposed by the choice of the basis of function:

.. math::

    &F^A_{P} (x, y)   =  \sqrt{2}\, \cos(P y), \\
    &F^K_{M,P} (x, y) =  2\cos(M nx)\, \sin(P y), \\
    &F^L_{H,P} (x, y) = 2\sin(H nx)\, \sin(P y)

which are specific Fourier mode respecting these boundary conditions, and where :math:`n` is the aspect ratio of the domain, i.e. the
ratio between the :math:`x`- and :math:`y`-extend of the domain.

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

2.2 Defining coordinates, parameters and fields of the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we are going to define some dimensional parameters for the model, using the dedicated :class:`~layercake.variables.parameter.Parameter` object.
To create a parameter, one need to pass it a value (which will be fixed), a |Sympy| symbol, and optionally its units:

.. code:: ipython3


    # Setting some parameters
    ##########################

    # Characteristic length scale (L_y / pi)
    L_symbol = Symbol('L')
    L = Parameter(1591549.4309189534, symbol=L_symbol, units='[m]')

    # Domain aspect ratio (this is a dimensionless parameter)
    n_symbol = Symbol('n')
    n = Parameter(1.3, symbol=n_symbol)

    # Coriolis parameter at the middle of the domain
    f0_symbol = Symbol('f_0')
    f0 = Parameter(1.032e-4, symbol=f0_symbol, units='[s^-1]')

    # Meridional gradient of the Coriolis parameter at phi_0
    beta_symbol = Symbol(u'β')
    beta = Parameter(1.3594204385792041e-11, symbol=beta_symbol, units='[m^-1][s^-1]')

    # Height of the atmospheric layers
    H1_symbol = Symbol('H_1')
    H1 = Parameter(5.2e3, symbol=H1_symbol, units='[m]')

    H2_symbol = Symbol('H_2')
    H2 = Parameter(5.2e3, symbol=H2_symbol, units='[m]')

    # Gravity
    g = Parameter(9.81, symbol=Symbol("g"), units='[m][s^-2]')

    # Reduced gravity
    gp = Parameter(float(g) * 0.6487, symbol=Symbol("g'"), units='[m][s^-2]')

We can now define derived non-dimensional parameters in the same way (note that we do not pass any :code:`units` argument here since these are dimensionless
parameters):

.. code:: ipython3

    # Derived (non-dimensional) parameters
    #######################################

    # Meridional gradient of the Coriolis parameter at phi_0
    beta_nondim = Parameter(beta * L / f0, symbol=beta_symbol)

    # Cross-terms
    alpha_1 = Parameter(f0 ** 2 * L ** 2 / (g * H1), symbol=Symbol("α_1"))
    alphap_1 = Parameter(f0 ** 2 * L ** 2 / (gp * H1), symbol=Symbol("α'_1"))
    dalpha_1 = Parameter(alpha_1 - alphap_1, symbol=Symbol("Δα_1"))
    alphap_2 = Parameter(f0 ** 2 * L ** 2 / (gp * H2), symbol=Symbol("α'_2"))

We can the move to the definition of the domain, using a dedicated function :func:`~layercake.basis.planar_fourier.contiguous_channel_basis` which
create a basis object :class:`~layercake.basis.planar_fourierPlanarChannelFourierBasis` with the right set of basis functions mentioned above.
The only parameter we need to pass to this function is the number of wavenumbers that we want in each direction, and the aspect ratio parameter :math:`n`
that we have previously defined:

.. code:: ipython3

    # Defining the domain
    ######################

    parameters = [n]
    atmospheric_basis = contiguous_channel_basis(2, 2, parameters)

Note that this function also create directly a :class:`~layercake.variables.systems.PlanarCartesianCoordinateSystem` object for you, representing
the :math:`x, y` coordinate system of the beta plane, and embedded in the :code:`atmospheric_basis` object.

We also need to create an inner product definition so that LayerCake knows how you want your PDEs to be projected.
In general, using the :class:`~layercake.inner_products.definition.StandardSymbolicInnerProductDefinition` definition is sufficient for most model:

.. code:: ipython3

    # creating a inner product definition with an optimizer for trigonometric functions
    inner_products_definition = StandardSymbolicInnerProductDefinition(coordinate_system=atmospheric_basis.coordinate_system,
                                                                       optimizer='trig', kwargs={'conds': 'none'})

Note how we passed directly the coordinate system embedded in the :code:`atmospheric_basis` object created before.
To simplify the writing of the code downstream, we can also save the symbols of the coordinate system now:

.. code:: ipython3

    # coordinates
    x = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[0]
    y = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[1]


References
----------

.. bibliography:: bib/ref.bib
    :labelprefix: USER-
    :keyprefix: user-

.. rubric:: Footnotes

.. [#basis] Note that noting prevent the users from using the same basis of function :math:`\phi_k` to decompose all the fields :math:`\psi_j`.

.. _Galerkin method: https://en.wikipedia.org/wiki/Galerkin_method
.. _orography: https://en.wikipedia.org/wiki/Orography
.. _beta coefficient: https://en.wikipedia.org/wiki/Beta_plane
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane