
"""
    Inner products definition module
    ================================

    Module containing classes to define the `inner products`_ used by the model.

    .. _inner products: https://en.wikipedia.org/wiki/Inner_product_space
    
"""

from abc import ABC, abstractmethod
from sympy.simplify.fu import TR8, TR10
from sympy import integrate, Integral


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
    """Standard class to define symbolic inner products using `Sympy`_.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.
    optimizer: None or callable, optional
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize.

    Attributes
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.
    optimizer: None or callable
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize the computation.

    .. _Sympy: https://www.sympy.org/

    """

    def __init__(self, coordinate_system, optimizer=None, kwargs=None):

        InnerProductDefinition.__init__(self)
        self.coordinate_system = coordinate_system

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

    @staticmethod
    def _trig_optimizer(expr):
        return TR10(TR8(expr))

    def integrate_over_domain(self, expr, symbolic_expr=False):
        """Definition of the normalized integrals over the spatial domain used by the inner products:
        :math:`\\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} \\, \\mathrm{expr}(x, y) \\, \\mathrm{d} x \\, \\mathrm{d} y`.

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
        _x = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        _y = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        _extent_x = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        _extent_y = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        if symbolic_expr:
            return Integral(expr, (_x, *_extent_x), (_y, *_extent_y), **self.kwargs)
        else:
            return integrate(expr, (_x, *_extent_x), (_y, *_extent_y), **self.kwargs)

    def inner_product(self, S, G, symbolic_expr=False, integrand=False):
        """Function defining the inner product to be computed symbolically:
        :math:`(S, G) = \\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} S(x,y)\\, G(x,y)\\, \\mathrm{d} x \\, \\mathrm{d} y`.

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
        _x = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[0]]
        _y = self.coordinate_system.coordinates_symbol[self.coordinate_system.coordinates_name[1]]
        _extent_x = self.coordinate_system.extent[self.coordinate_system.coordinates_name[0]]
        _extent_y = self.coordinate_system.extent[self.coordinate_system.coordinates_name[1]]
        norm = ((_extent_x[1] - _extent_x[0]) * (_extent_y[1] - _extent_y[0]))
        expr = (S * G) / norm
        if integrand:
            return expr,  (_x, *_extent_x), (_y, *_extent_y)
        else:
            return self.integrate_over_domain(self.optimizer(expr), symbolic_expr=symbolic_expr)
