"""
    Basis definition module (base class)
    ====================================

    Abstract base classes defining the functions (modes) of the basis of the model and used to configure it.

    Description of the classes
    --------------------------

    * :class:`Basis`: General base class.
    * :class:`SymbolicBasis`: Base class for symbolic functions basis.

    Warnings
    --------

    These are `abstract base class`_, they must be subclassed to create new basis!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

"""

import sys

from abc import ABC
from sympy import Symbol, lambdify, diff


class Basis(ABC):
    """General base class for a basis of functions.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.

    Attributes
    ----------
    functions: list
        List of functions of the basis.
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.
    """

    def __init__(self, coordinate_system):

        self.functions = list()
        self.coordinate_system = coordinate_system

    def __getitem__(self, index):
        return self.functions[index]

    def __repr__(self):
        return self.functions.__repr__()

    def __str__(self):
        return self.functions.__str__()

    def __len__(self):
        return self.functions.__len__()

    def append(self, item):
        self.functions.append(item)


class SymbolicBasis(Basis):
    """General base class for a basis of symbolic functions.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.
    parameters: dict(~parameter.Parameter)
        Dictionary holding the parameters appearing in the equations defining the basis.

    Attributes
    ----------
    substitutions: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions. The 2-tuples contain first
        a `Sympy`_  expression and then the value to substitute.
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system on which the basis is defined.
    parameters: list(~parameter.Parameter)
        Dictionary holding the parameters appearing in the equations defining the basis.

    .. _Sympy: https://www.sympy.org/

    """

    def __init__(self, coordinate_system, parameters):

        Basis.__init__(self, coordinate_system)
        self.substitutions = list()
        self.parameters = parameters

    @property
    def parameters_symbols(self):
        return [p.symbol for p in self.parameters]

    def set_parameters(self, parameters):
        """Setter for the `parameters` attributes.
        Must be redefined for subclasses.

        Parameters
        ----------
        parameters: dict(~parameter.Parameter)
            Dictionary holding the parameters appearing in the equations defining the basis.
        """
        self.parameters = parameters

    def subs_functions(self, extra_subs=None):
        """Return the basis functions with the substitutions stored in the object being applied.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions. The 2-tuples contain first
            a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list
            List of the substituted basis functions
        """

        sf = list()

        for f in self.functions:
            if extra_subs is not None:
                ff = f.subs(extra_subs)
            else:
                ff = f
            ff = ff.subs(self.substitutions)
            sf.append(ff)

        return sf

    def num_functions(self, extra_subs=None):
        """Return the basis functions with as python callable.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions before transforming them into
            python callable. The 2-tuples contain first a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list(callable)
            List of callable basis functions
        """

        coordinates_symbol = self.coordinate_system.coordinates_symbol

        nf = list()
        sf = self.subs_functions(extra_subs=extra_subs)

        for f in sf:
            try:
                nf.append(lambdify(coordinates_symbol, f))
            except:
                tb = sys.exc_info()[2]
                raise Exception.with_traceback(tb)

        return nf

    def derivative(self, symbol, order=1):
        """Return the basis functions differentiated with respect to `symbol` as a new basis.

        Parameters
        ----------
        symbol: Sympy symbol
            The symbol with respect to which the basis is to be differentiated.
        order: int, optional
            The order of the derivative. Default to first order.

        Returns
        -------
        SymbolicBasis:
            A new basis object with the differentiated basis function.
        """

        dfunc = list(map(lambda func: diff(func, symbol, order), self.functions))
        dbasis = SymbolicBasis(self.coordinate_system)
        dbasis.functions = dfunc
        dbasis.substitutions = self.substitutions

        return dbasis

    def directional_derivative(self, order=1):
        """Return the basis functions differentiated with respect to the coordinates.

        Parameters
        ----------
        order: int, optional
            The order of the derivative. Default to first order.

        Returns
        -------
        SymbolicBasis:
            A new basis object with the differentiated basis function.
        """
        return {name: self.derivative(variable, order)
                for name, variable in zip(self.coordinate_system.coordinates_name,
                                          self.coordinate_system.coordinates_symbol_as_list)}


# Rem: Class not used currently in the model.
class NumericBasis(Basis):
    """General base class for a basis of numeric functions.

    """

    def __init__(self):

        Basis.__init__(self)

    def num_functions(self):
        """Return the basis functions with as python callable.

        Returns
        -------
        list(callable)
            List of callable basis functions
        """

        return self.functions


if __name__ == "__main__":
    import numpy as np
    from sympy import symbols, sin, exp
    from layercake.variables.coordinate import Coordinate
    from layercake.variables.systems import CoordinateSystem

    xs, ys = symbols('x y')  # x and y coordinates on the model's spatial domain
    x = Coordinate("x", xs, extent=[0., 2 * np.pi])
    y = Coordinate("y", ys, extent=[0., np.pi])
    coord_sys = CoordinateSystem([x, y])
    basis = SymbolicBasis(coord_sys)
    al = symbols('al')  # aspect ratio and alpha coefficients
    n = Symbol('n', positive=True)
    for i in range(1, 3):
        for j in range(1, 3):
            basis.append(2 * exp(- al * xs) * sin(j * n * xs / 2) * sin(i * ys))

    basis.substitutions.append(('n', 1.))
    basis.substitutions.append(('al', 1.))
  
