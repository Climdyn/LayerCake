
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

symbolic_computation_timeout = 200


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
    optimizer: None or callable
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize the computation.

    """

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
    def _trig_optimizer(expr):
        res = list()
        res.append(_matching(expr))
        res.append(_old(expr))
        res.append(_fu(expr))

        measure = list()
        for i, r in enumerate(res):
            try:
                measure.append(len(str(r)))
            except AttributeError:
                res[i] = None
                measure.append(1000000000+i)
        print(f'selecting {measure.index(min(measure))}')
        sel_res = res[measure.index(min(measure))]
        if sel_res is None:
            raise TimeoutError(f'Simplification of symbolic expression in integrals: No simplifications '
                               f'were achieved in less that {symbolic_computation_timeout} seconds !')
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
        _u = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        _v = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        _extent_u = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        _extent_v = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        if symbolic_expr:
            return Integral(expr, (_u, *_extent_u), (_v, *_extent_v), **self.kwargs)
        else:
            return integrate(expr, (_u, *_extent_u), (_v, *_extent_v), **self.kwargs)

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
        _u = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        _v = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        _u_elem = self.coordinate_system.coordinates[0].infinitesimal_length
        _v_elem = self.coordinate_system.coordinates[1].infinitesimal_length
        _extent_u = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        _extent_v = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        norm = ((_extent_u[1] - _extent_u[0]) * (_extent_v[1] - _extent_v[0]))
        print(f'')
        print(f'S={S}')
        print(f'G={G}')
        print(f'fu(G)={self.optimizer(G)}')
        print(f'')
        if self.complex:
            expr = (S * conjugate(G)) / norm
        else:
            expr = (S * G) / norm
        if integrand:
            return expr * _u_elem * _v_elem,  (_u, *_extent_u), (_v, *_extent_v)
        else:
            return self.integrate_over_domain(self.optimizer(expr * _u_elem * _v_elem), symbolic_expr=symbolic_expr)


@exit_after(symbolic_computation_timeout)
def _fu(expr):
    return trigsimp(expr, method='fu')


@exit_after(symbolic_computation_timeout)
def _matching(expr):
    return trigsimp(expr, method='matching')


@exit_after(symbolic_computation_timeout)
def _old(expr):
    return trigsimp(expr, method='old')
