"""
    Parameter module
    ================

    This module contains the basic parameter class to hold model's parameters values.
    It allows to manipulate dimensional and nondimensional parameter easily.

    Examples
    --------

    >>> from layercake.variables.parameter import Parameter, ParametersArray
    >>> import numpy as np
    >>> # creating a parameter initialized with a nondimensional value
    >>> sigma = Parameter(0.2e0,
    ...                   description="static stability of the atmosphere (nondimensional)")
    >>> sigma
    0.2
    >>> # creating a parameter initialized with a dimensional value
    >>> sigma = Parameter(2.1581898457499433e-06,
    ...                   units='[m^2][s^-2][Pa^-2]',
    ...                   description="static stability of the atmosphere")
    >>> sigma
    2.1581898457499433e-06
    >>> # creating a parameters array initialized with a nondimensional values
    >>> s = ParametersArray(np.array([[0.1,0.2],[0.3,0.4]]), units='',
    ...                     description="atmosphere bottom friction coefficient (nondimensional")
    >>> s
    ArrayParameters([[0.1, 0.2],
                     [0.3, 0.4]], dtype=object)
    >>> # you can also ask for the value of one particular value of the array
    >>> s[0,0]
    0.1

    Main class
    ----------
"""

import warnings
import numpy as np
from fractions import Fraction

from layercake.variables.utils import combine_units


# TODO: Automatize warnings and errors
# TODO: Implement operations for arrays

class Parameter(float):
    """Base class of model's parameter.

    Parameters
    ----------
    value: float
        Value of the parameter.
    units: str, optional
        The units of the provided value.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    description: str, optional
        String describing the parameter.
    symbol: ~sympy.core.symbol.Symbol, optional
        A `Sympy`_ symbol to represent the parameter in symbolic expressions.
    symbolic_expression: ~sympy.core.expr.Expr, optional
        A `Sympy`_ expression to represent a relationship to other parameters.

    Notes
    -----
    Parameter is immutable. Once instantiated, it cannot be altered. To create a new parameter, one must
    re-instantiate it.

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, value, units="", description="", symbol=None, symbolic_expression=None):

        no_scale = False

        f = float.__new__(cls, value)
        f._units = units
        f._description = description
        f._symbol = symbol
        f._symbolic_expression = symbolic_expression

        return f

    @property
    def symbol(self):
        """~sympy.core.symbol.Symbol: Returns the symbol of the parameter."""
        return self._symbol

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Returns the symbolic expression of the parameter."""
        if self._symbolic_expression is None and self._symbol is not None:
            return self._symbol
        else:
            return self._symbolic_expression
    
    @property
    def units(self):
        """str: The units of the dimensional value."""
        return self._units

    @property
    def description(self):
        """str: Description of the parameter."""
        return self._description

    def __add__(self, other):

        res = float(self) + other
        if isinstance(other, Parameter):
            if self.units != other._units:
                raise ArithmeticError("Parameter class: Impossible to add two parameters with different units.")

            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol + other.symbol
                    else:
                        expr = None
                    descr = self.description + " + " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol + (other.symbolic_expression)
                        descr = self.description + " + (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) + other.symbol
                        descr = "(" + self.description + ") + " + other.description
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
                else:
                    expr = (self.symbolic_expression) + (other.symbolic_expression)
                    descr = "(" + self.description + ") + (" + other.description + ")"

            return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol + other
                    descr = self.description + " + " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) + other
                    descr = "(" + self.description + ") + " + str(other)
                else:
                    expr = None
                    descr = self.description + " + " + str(other)
                return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        res = float(self) - other
        if isinstance(other, Parameter):
            if self.units != other._units:
                raise ArithmeticError("Parameter class: Impossible to subtract two parameters with different units.")
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol - other.symbol
                    else:
                        expr = None
                    descr = self.description + " - " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol - (other.symbolic_expression)
                        descr = self.description + " - (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) - other.symbol
                        descr = "(" + self.description + ") - " + other.description
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
                else:
                    expr = (self.symbolic_expression) - (other.symbolic_expression)
                    descr = "(" + self.description + ") - (" + other.description + ")"

            return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol - other
                    descr = self.description + " - " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) - other
                    descr = "(" + self.description + ") - " + str(other)
                else:
                    expr = None
                    descr = self.description + " - " + str(other)
                return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rsub__(self, other):
        res = other - float(self)
        try:
            if self.symbol is not None:
                expr = other - self.symbol
                descr = str(other) + " - " + self.description
            elif self.symbolic_expression is not None:
                expr = other - (self.symbolic_expression)
                descr = str(other) + " - (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " - " + self.description
            return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __mul__(self, other):

        res = float(self) * other
        if isinstance(other, Parameter):
            if hasattr(other, "units"):
                units = combine_units(self.units, other._units, '+')
            else:
                units = ""

            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol * other.symbol
                    else:
                        expr = None
                    descr = self.description + " * " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol * (other.symbolic_expression)
                        descr = self.description + " * (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) * other.symbol
                        descr = "(" + self.description + ") * " + other.description
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
                else:
                    expr = (self.symbolic_expression) * (other.symbolic_expression)
                    descr = "(" + self.description + ") * (" + other.description + ")"

            return Parameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol * other
                    descr = self.description + " * " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) * other
                    descr = "(" + self.description + ") * " + str(other)
                else:
                    expr = None
                    descr = self.description + " * " + str(other)
                return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        res = float(self) / other
        if isinstance(other, Parameter):
            units = combine_units(self.units, other._units, '-')
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol / other.symbol
                    else:
                        expr = None
                    descr = self.description + " / " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol / (other.symbolic_expression)
                        descr = self.description + " / (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) / other.symbol
                        descr = "(" + self.description + ") / " + other.description
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
                else:
                    expr = (self.symbolic_expression) / (other.symbolic_expression)
                    descr = "(" + self.description + ") / (" + other.description + ")"

            return Parameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol / other
                    descr = self.description + " / " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) / other
                    descr = "(" + self.description + ") / " + str(other)
                else:
                    expr = None
                    descr = self.description + " / " + str(other)
                return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rtruediv__(self, other):
        res = other / float(self)
        try:
            if self.symbol is not None:
                expr = other / self.symbol
                descr = str(other) + " / " + self.description
            elif self.symbolic_expression is not None:
                expr = other / (self.symbolic_expression)
                descr = str(other) + " / (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " / " + self.description
            return Parameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __pow__(self, power, modulo=None):

        if modulo is not None:
            raise NotImplemented('Parameter class: Modular exponentiation not implemented')

        res = float(self) ** power
        if int(power) == power:

            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                units_elements.append(list((us[0], str(int(us[1]) * power))))

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)

            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power "+str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power "+str(power)
            else:
                expr = None
                descr = self.description + " to the power "+str(power)

        else:
            power_fraction = Fraction(power)
            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                new_power = int(us[1]) * power_fraction.numerator / power_fraction.denominator
                if int(new_power) == new_power:
                    units_elements.append(list((us[0], str(int(new_power)))))
                else:
                    raise ArithmeticError("Parameter class: Only support integer exponent in units")

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)
            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power "+str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power "+str(power)
            else:
                expr = None
                descr = self.description + " to the power "+str(power)

        return Parameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)


class ParametersArray(np.ndarray):
    """Base class of model's array of parameters.

    Parameters
    ----------
    values: list(float) or ~numpy.ndarray(float) or list(Parameter) or ~numpy.ndarray(Parameter)
        Values of the parameter array.
    units: str, optional
        The units of the provided value.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    description: str or list(str) or array(str), optional
        String or an iterable of strings, describing the parameters.
        If an iterable, should have the same length or shape as `values`.
    symbols ~sympy.core.symbol.Symbol or list(~sympy.core.symbol.Symbol) or ~numpy.ndarray(~sympy.core.symbol.Symbol), optional
        A `Sympy`_ symbol or an iterable of symbols, to represent the parameters in symbolic expressions.
        If an iterable, should have the same length or shape as `values`.
    symbolic_expressions: ~sympy.core.expr.Expr or list(~sympy.core.expr.Expr) or ~numpy.ndarray(~sympy.core.expr.Expr), optional
        A `Sympy`_ expression or an iterable of expressions, to represent a relationship to other parameters.
        If an iterable, should have the same length or shape as `values`.

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, values, units="", description="", symbols=None, symbolic_expressions=None):

        if isinstance(values, (tuple, list)):
            new_arr = np.empty(len(values), dtype=object)
            for i, val in enumerate(values):
                if isinstance(description, (tuple, list, np.ndarray)):
                    descr = description[i]
                else:
                    descr = description
                if isinstance(symbols, (tuple, list, np.ndarray)):
                    sy = symbols[i]
                else:
                    sy = symbols
                if isinstance(symbolic_expressions, (tuple, list, np.ndarray)):
                    expr = symbolic_expressions[i]
                else:
                    expr = symbolic_expressions
                new_arr[i] = Parameter(val, units=units, description=descr, symbol=sy, symbolic_expression=expr)
        else:
            if isinstance(values.flatten()[0], Parameter):
                new_arr = values.copy()
            else:
                new_arr = np.empty_like(values, dtype=object)
                for idx in np.ndindex(values.shape):
                    if isinstance(description, np.ndarray):
                        descr = description[idx]
                    else:
                        descr = description
                    if isinstance(symbols, np.ndarray):
                        sy = symbols[idx]
                    else:
                        sy = symbols
                    if isinstance(symbolic_expressions, np.ndarray):
                        expr = symbolic_expressions[idx]
                    else:
                        expr = symbolic_expressions
                    new_arr[idx] = Parameter(values[idx], units=units, description=descr, symbol=sy, symbolic_expression=expr)
        arr = np.asarray(new_arr).view(cls)
        arr._units = units

        return arr

    def __array_finalize__(self, arr):

        if arr is None:
            return

        self._units = getattr(arr, '_units', "")

    @property
    def symbols(self):
        """~numpy.ndarray(~sympy.core.symbol.Symbol): Returns the symbol of the parameters in the array."""
        symbols = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            symbols[idx] = self[idx].symbol
        return symbols

    @property
    def symbolic_expressions(self):
        """~numpy.ndarray(~sympy.core.expr.Expr): Returns the symbolic expressions of the parameters in the array."""
        symbolic_expressions = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            symbolic_expressions[idx] = self[idx].symbolic_expression
        return symbolic_expressions


    @property
    def units(self):
        """str: The units of the dimensional value."""
        return self._units

    @property
    def descriptions(self):
        """~numpy.ndarray(str): Description of the parameters in the array."""
        descr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            descr[idx] = self[idx].description
        return descr

    def __add__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] + other
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] + other[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return self + other
        else:
            return self + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] - other
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] - other[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return self - other
        else:
            return self - other

    def __rsub__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = other - self[idx]
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = other - self[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return other - self
        else:
            return other - self

    def __mul__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] * other
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] * other[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return self * other
        else:
            return self * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] / other
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] / other[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return self / other
        else:
            return self / other

    def __rtruediv__(self, other):
        if isinstance(other, (Parameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = other / self[idx]
            item = res[idx]
            return ParametersArray(res, units=item.units)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = other / self[idx]
                item = res[idx]
                return ParametersArray(res, units=item.units)
            else:
                return other / self
        else:
            return other / self


