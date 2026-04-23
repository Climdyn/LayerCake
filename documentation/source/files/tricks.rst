
Various tricks
==============

On this page, we list several tricks on how to do this or that, related to particular aspects
of the modelling based on systems of partial differential equations.

Spatially varying parameters
----------------------------

It is possible in LayerCake to add parameters varying as a function of the coordinates.
To illustrate this, we are going to modify a spatially uniform parameter in one of the
`examples <../../../../examples/>`_, namely the
`Reinhold & Pierrehumber model <https://github.com/Climdyn/LayerCake/blob/main/examples/atmospheric/baroclinic_one_layer.py>`_ one.

In this model, two terms are representing the friction of the bottom layer with the ground:

* :math:`-\frac{k_d}{2} \nabla^2 (\psi - \theta)` for the equation involving the barotropic streamfunction :math:`\psi`
* :math:`+\frac{k_d}{2} \nabla^2 (\psi - \theta)` for the equation involving the baroclinic streamfunction :math:`\theta`

where the parameter controlling the friction is :math:`k_d`. The full equations of the model can be found `here <https://qgs.readthedocs.io/en/latest/files/model/oro_model.html#mid-layer-equations-and-the-thermal-wind-relation>`_.

Running the `LayerCake code <../../../../examples/atmospheric/baroclinic_one_layer.py>`_  produces the following figure:

.. figure:: tricks/RP1982.png
    :scale: 100%
    :align: center

which is a section of the model's attractor in the :math:`\psi_2, \psi_3` plane.

Now we can modify the original model's equations by making the parameter :math:`k_d` varies as

.. math::

    k_{d,0} + 2 \, k_{d,2} \cos(n x) \sin(y)

This can be done by using the :class:`~layercake.arithmetic.terms.operations.ProductOfTerms` objects to do the product of
a :class:`~layercake.arithmetic.terms.linear.LinearTerm` term involving a :class:`~layercake.variables.field.ParameterField` field representing
the spatial variation of the parameter, and the Laplacian terms representing :math:`\nabla^2 (\psi - \theta)`.
It can be implemented with a few lines added to the `Reinhold & Pierrehumber model code <https://github.com/Climdyn/LayerCake/blob/main/examples/atmospheric/baroclinic_one_layer.py>`_:

.. code:: ipython3

    dfriction = ProductOfTerms(LinearTerm(Dk), OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system, sign=-1))
    dofriction = ProductOfTerms(LinearTerm(Dk), OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system))
    barotropic_equation.add_rhs_terms([dfriction, dofriction])

in the barotropic equation and

.. code:: ipython3

    dfriction = ProductOfTerms(LinearTerm(Dk), OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system, sign=1))
    dofriction = ProductOfTerms(LinearTerm(Dk), OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system, sign=-1))
    baroclinic_equation.add_rhs_terms([dfriction, dofriction])

in the baroclinic equation, where :code:`Dk` is the :class:`~layercake.variables.field.ParameterField` object defining the spatial
variation of :math:`k_d`

.. code:: ipython3

    # Variable bottom friction
    dk = np.zeros(len(atmospheric_basis))
    dk[1] = 0.1 * kdp_deriv
    Dk = ParameterField('D_k', u'D_k', dk, atmospheric_basis, inner_products_definition)

with :math:`k_{d,2} = D_{k,1} = 0.1 \, k_d` (note the difference of index because of the Python-specific indexing starting from zero).
The equations LaTeX representation now clearly shows the new terms:

.. figure:: tricks/mod_eqs.png
    :align: center

with the appearance of the :math:`D_k \nabla^2` terms.

The impact of this spatial variation of the bottom friction coefficient on the model's dynamics
is clearly visible on the 2-dimensional section of the attractor:

.. figure:: tricks/RP1982mod.png
    :scale: 100%
    :align: center

(Compare with the first figure above.)

.. Section below should be revisited when a proper example exists
.. Usage of mathematical expressions in the PDEs
.. ---------------------------------------------

.. Mathematical function can appear in some models PDEs. In general, these are functions of the model's domain coordinates.
.. For example, in the `stratospheric example model <https://github.com/Climdyn/LayerCake/blob/main/examples/atmospheric/stratospheric_planetary_flow.py>`_
..present in the `atmospheric examples folder <../../../../examples/atmospheric/>`_, a

Free-threading
--------------

Since Python 3.14, a `free-threaded version of Python is available <https://docs.python.org/3/howto/free-threading-python.html>`_.
LayerCake has been tested for it and seems so far to run smoothly.

This feature is particularly useful if you try to derive complicate, big models with a
large number of modes. LayerCake will then use multiple threads to perform |Sympy| symbolic evaluations,
while the integration of the inner products will still be done using multiple processes.
When being used in a free-threading environment, this is the default behavior, but this can be controlled by environment
variables:

* The :code:`LAYERCAKE_PARALLEL_METHOD` environment variable defines how the Sympy symbolic evaluations are done. It can take two different values:
    + :code:`threads`: the evaluations will be done using threads
    + :code:`processes`: the evaluations will be done using processes. In some complicate cases, it might lead to wrong answers or even crash. This mode is thus not recommended unless you know what you are doing.
  If this environment variable is not defined, then LayerCake default behavior is to use threads.
* The :code:`LAYERCAKE_PARALLEL_INTEGRATION` environment variable controls the Sympy symbolic integration parallelization. If set to :code:`none`, the parallelization will be deactivated.
  Otherwise, it will parallelized using processes.
  If this environment variable is not defined, then LayerCake default behavior is to parallelize using processes.

For example,

.. code:: bash

    LAYERCAKE_PARALLEL_METHOD=processes python examples/atmospheric/barotropic_one_layer.py

will launch the <https://github.com/Climdyn/LayerCake/blob/main/examples/atmospheric/barotropic_one_layer.py>`_
script with Sympy symbolic evaluation being done using processes.

Installing the free-threaded version of Python can be done using Anaconda, by typing:

.. code:: bash

    conda env create -f environment_freethreading.yml

which will create a conda environment :code:`layercake_ft`.
Upon activation:

.. code:: bash

    conda activate layercake_ft

any Python code will be run in `free-threading` mode.

.. warning::

    Python free-threading mode is still somewhat experimental, and
    the obtained models and results must be
    scrutinized with care and double-checked.

