
"""
    Inner products definition module
    ================================

    Module containing classes to define the `inner products`_ used by the model.

    .. _inner products: https://en.wikipedia.org/wiki/Inner_product_space

    Main classes
    ------------

"""

from abc import ABC, abstractmethod
# from sympy.simplify.fu import TR8, TR10  # old qgs optimizer functions
from sympy import trigsimp
from sympy import integrate, Integral, conjugate

from layercake.utils.parallel import exit_after


class InnerProductDefinition(ABC):
    """Base class to define the model's basis inner products."""

    def __init__(self):
        pass

    @abstractmethod
    def inner_product(self, S, G):
        """Definition of the inner product :math:`(S, G)`.

        Parameters
        ----------
        S:
            Left-hand side function of the product.
        G:
            Right-hand side function of the product.

        Returns
        -------
        res
            The result of the inner product.
        """
        pass


class StandardSymbolicInnerProductDefinition(InnerProductDefinition):
    """Standard class to define symbolic inner products using |Sympy|.

    Parameters
    ----------
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.
    optimizer: None or callable or str, optional
        A function to optimize the computation of the integrals or the integrand.
        If a string, specifies pre-defined optimizers:
        * `'trig'`: Optimizer specifically designed for trigonometric functions.
        If `None`, does not optimize.
    complex: bool, optional
        Whether to compute the inner products with complex conjugate expression
        for the second term.
        Default to `False`, i.e. real inner products.
    kwargs: dict
        Specific keywords arguments to pass to the |Sympy| integrals,
        see :func:`~sympy.integrals.integrals.integrate` and
        :class:`~sympy.integrals.integrals.Integral`.

    Attributes
    ----------
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.
    complex: bool, optional
        Whether to compute the inner products with complex conjugate expression
        for the second term.
    optimizer: None or callable
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize the computation.

    """

    symbolic_computation_timeout = 200  # default value

    def __init__(self, coordinate_system, optimizer=None, complex=False, kwargs=None):

        InnerProductDefinition.__init__(self)
        self.coordinate_system = coordinate_system
        self.complex = complex

        if optimizer is None:
            self.optimizer = self._no_optimizer
        elif optimizer == 'trig':
            self.optimizer = self._trig_optimizer
        else:
            self.optimizer = optimizer

        if kwargs is not None:
            self.kwargs = kwargs
        else:
            self.kwargs = dict()

    def set_optimizer(self, optimizer):
        """Function to set the optimizer.

        Parameters
        ----------
        optimizer: callable
            A function to optimize the computation of the integrals or the integrand.
        """
        self.optimizer = optimizer

    @staticmethod
    def _no_optimizer(expr):
        return expr

    # @staticmethod
    # def _trig_optimizer(expr):  # old qgs optimizer
    #     return TR10(TR8(expr))

    @staticmethod
    @exit_after(symbolic_computation_timeout)
    def _fu(expr):
        return trigsimp(expr, method='fu')

    @staticmethod
    @exit_after(symbolic_computation_timeout)
    def _matching(expr):
        return trigsimp(expr, method='matching')

    @staticmethod
    @exit_after(symbolic_computation_timeout)
    def _old(expr):
        return trigsimp(expr, method='old')

    @classmethod
    def _trig_optimizer(cls, expr):
        res = list()
        res.append(cls._matching(expr))
        res.append(cls._old(expr))
        res.append(cls._fu(expr))

        measure = list()
        for i, r in enumerate(res):
            try:
                measure.append(len(str(r)))
            except AttributeError:
                res[i] = None
                measure.append(1000000000+i)
        sel_res = res[measure.index(min(measure))]
        if sel_res is None:
            raise TimeoutError(f"Simplification of symbolic expression in integrals: No simplifications "
                               f"were achieved in less that {cls.symbolic_computation_timeout} seconds !"
                               f" Change the 'symbolic_computation_timeout' attribute of the inner product"
                               f" definition class if you want to try longer computation time.")
        return sel_res

    def integrate_over_domain(self, expr, symbolic_expr=False):
        """Definition of the integrals over the spatial domain used by the inner products:
        :math:`\\int_a^b\\int_c^d \\, \\mathrm{expr}(x, y) \\, \\mathrm{d} x \\, \\mathrm{d} y`.

        Parameters
        ----------
        expr: ~sympy.core.expr.Expr
            The expression to integrate.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        ~sympy.core.expr.Expr
            The result of the symbolic integration.
        """
        u = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        v = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        extent_u = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        extent_v = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        if symbolic_expr:
            return Integral(expr, (u, *extent_u), (v, *extent_v), **self.kwargs)
        else:
            return integrate(expr, (u, *extent_u), (v, *extent_v), **self.kwargs)

    def inner_product(self, S, G, symbolic_expr=False, integrand=False):
        """Function defining the inner product to be computed symbolically:
        :math:`(S, G) = \\left(1 / \\mathcal{N}\\right) \\int_a^b\\int_c^d S(x,y)\\, G(x,y)\\, \\mathrm{d} x \\, \\mathrm{d} y` where
        :math:`\\mathcal{N} = (b-a) \\, (d-c)` is the norm of the integrals.

        Parameters
        ----------
        S: ~sympy.core.expr.Expr
            Left-hand side function of the product.
        G: ~sympy.core.expr.Expr
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        ~sympy.core.expr.Expr
            The result of the symbolic integration
        """
        u = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        v = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        u_elem = self.coordinate_system.coordinates[0].infinitesimal_length
        v_elem = self.coordinate_system.coordinates[1].infinitesimal_length
        extent_u = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        extent_v = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        norm = ((extent_u[1] - extent_u[0]) * (extent_v[1] - extent_v[0]))
        mod_G = self.optimizer(G)
        if self.complex:
            expr = (S * conjugate(mod_G)) / norm
        else:
            expr = (S * mod_G) / norm
        if integrand:
            return expr * u_elem * v_elem,  (u, *extent_u), (v, *extent_v)
        else:
            return self.integrate_over_domain(self.optimizer(expr * u_elem * v_elem), symbolic_expr=symbolic_expr)
