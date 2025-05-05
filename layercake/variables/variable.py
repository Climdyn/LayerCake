
from abc import ABC
from sympy import Symbol
import numpy as np


class Variable(ABC):

    def __init__(self, name, symbol, units=None, latex=None):

        self.name = name
        if isinstance(symbol, str):
            self.symbol = Symbol(symbol)
        else:
            self.symbol = symbol
        if units is None:
            self.units = ""
        else:
            self.units = units

        if latex is None:
            self.latex = name
        else:
            self.latex = latex

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ')'

    def __repr__(self):
        return self.__str__()


class VariablesArray(np.ndarray):
    """Base class of model's array of variables values.

    Parameters
    ----------
    values: list(float) or ~numpy.ndarray(float)
        Values of the variables array.
    name: str
        General name of the variables.
    symbol: str or ~sympy.core.symbol.Symbol
        A `Sympy`_ symbol to represent the variables in symbolic expressions.
    units: str, optional
        The units of the provided value. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        A latex string representing the variables. Used in plots.
        Empty by default.
    scale_object: ScaleParams, optional
        A scale parameters object to compute the conversion between dimensional and nondimensional value.
        `None` by default. If `None`, cannot transform between dimensional and nondimentional value.
    input_dimensional: bool, optional
        Specify whether the value provided is dimensional or not. Default to `True`.
    return_dimensional: bool, optional
        Defined if the value returned by the variables is dimensional or not. Default to `False`.

    Warnings
    --------
    If no scale_object argument is provided, cannot transform between the dimensional and nondimensional value !

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, values, name, symbol, units="", latex=None, scale_object=None,
                input_dimensional=False, return_dimensional=False):

        if not isinstance(symbol, str):
            symbol = symbol.name

        symbols = [f'{symbol}_{i}' for i in range(len(values))]
        names = [f'{name}_{i}' for i in range(len(values))]
        if latex is not None:
            latexes = [f'{latex}_{i}' for i in range(len(values))]
        else:
            latexes = None
        if isinstance(values, np.ndarray):
            new_arr = values.copy()
        else:
            new_arr = np.array(values)
        arr = np.asarray(new_arr).view(cls)
        arr._names = names
        arr._name = name
        arr._symbols = symbols
        arr._symbol = Symbol(symbol)
        arr._latex = latex
        arr._latexes = latexes
        arr._input_dimensional = input_dimensional
        arr._return_dimensional = return_dimensional
        arr._units = units
        arr._scale_object = scale_object

        return arr

    def __array_finalize__(self, arr):

        if arr is None:
            return

        self._names = getattr(arr, '_names', [''] * len(arr))
        self._name = getattr(arr, '_name', [''])
        self._symbols = getattr(arr, '_symbols', np.zeros(len(arr), dtype=object))
        self._symbol = getattr(arr, '_symbol', None)
        self._latex = getattr(arr, '_latex', None)
        self._latexes = getattr(arr, '_latexes', None)
        self._input_dimensional = getattr(arr, '_input_dimensional', True)
        self._units = getattr(arr, '_units', "")
        self._return_dimensional = getattr(arr, '_return_dimensional', False)
        self._scale_object = getattr(arr, '_scale_object', None)

    @property
    def dimensional_values(self):
        """float: Returns the dimensional value."""
        if self._return_dimensional:
            return self
        else:
            return self / self._nondimensionalization

    @property
    def nondimensional_values(self):
        """float: Returns the nondimensional value."""
        if self._return_dimensional:
            return self * self._nondimensionalization
        else:
            return self

    @property
    def symbol(self):
        """~sympy.core.symbol.Symbol: Returns the general symbol of the variables in the array."""
        return self._symbol

    @property
    def symbols(self):
        """~numpy.ndarray(~sympy.core.symbol.Symbol): Returns the symbols of the variables in the array."""
        return self._symbols

    @property
    def latex(self):
        """str: Returns the general latex expression of the variables in the array."""
        return self._latex

    @property
    def latexes(self):
        """str: Returns the latex expressions of the variables in the array."""
        return self._latexes

    @property
    def name(self):
        """str: Returns the general name of the variables in the array."""
        return self._name

    @property
    def names(self):
        """str: Returns the names of the variables in the array."""
        return self._names

    @property
    def input_dimensional(self):
        """bool: Indicate if the provided value is dimensional or not."""
        return self._input_dimensional

    @property
    def return_dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self._return_dimensional

    @classmethod
    def _conversion_factor(cls, units, scale_object):
        factor = 1.

        ul = units.split('][')
        ul[0] = ul[0][1:]
        ul[-1] = ul[-1][:-1]

        for us in ul:
            up = us.split('^')
            if len(up) == 1:
                up.append("1")

            if up[0] == 'm':
                factor *= scale_object.L ** (-int(up[1]))
            elif up[0] == 's':
                factor *= scale_object.f0 ** (int(up[1]))
            elif up[0] == 'Pa':
                factor *= scale_object.deltap ** (-int(up[1]))

        return factor

    @property
    def units(self):
        """str: The units of the dimensional values."""
        return self._units

    @property
    def _nondimensionalization(self):
        if self._scale_object is None:
            return 1.
        else:
            return self._conversion_factor(self._units, self._scale_object)

    #   Not sure these operations will be needed.
    #
    # def __add__(self, other):
    #     if isinstance(other, VariablesArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             if self.units != other.units:
    #                 raise ValueError('Cannot add the provided VariablesArray because their units are different.')
    #             res = self + other
    #             return VariablesArray(res,
    #         else:
    #             return self + other
    #     else:
    #         return self + other
    #
    # def __radd__(self, other):
    #     return self.__add__(other)
    #
    # def __sub__(self, other):
    #     if isinstance(other, (Parameter, ScalingParameter, float, int)):
    #         res = np.empty(self.shape, dtype=object)
    #         for idx in np.ndindex(self.shape):
    #             res[idx] = self[idx] - other
    #         item = res[idx]
    #         return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                units=item.units, scale_object=self._scale_object)
    #     elif isinstance(other, ParametersArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             res = np.empty(self.shape, dtype=object)
    #             for idx in np.ndindex(self.shape):
    #                 res[idx] = self[idx] - other[idx]
    #             item = res[idx]
    #             return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                    units=item.units, scale_object=self._scale_object)
    #         else:
    #             return self - other
    #     else:
    #         return self - other
    #
    # def __rsub__(self, other):
    #     if isinstance(other, (Parameter, ScalingParameter, float, int)):
    #         res = np.empty(self.shape, dtype=object)
    #         for idx in np.ndindex(self.shape):
    #             res[idx] = other - self[idx]
    #         item = res[idx]
    #         return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                units=item.units, scale_object=self._scale_object)
    #     elif isinstance(other, ParametersArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             res = np.empty(self.shape, dtype=object)
    #             for idx in np.ndindex(self.shape):
    #                 res[idx] = other - self[idx]
    #             item = res[idx]
    #             return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                    units=item.units, scale_object=self._scale_object)
    #         else:
    #             return other - self
    #     else:
    #         return other - self
    #
    # def __mul__(self, other):
    #     if isinstance(other, (Parameter, ScalingParameter, float, int)):
    #         res = np.empty(self.shape, dtype=object)
    #         for idx in np.ndindex(self.shape):
    #             res[idx] = self[idx] * other
    #         item = res[idx]
    #         return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                units=item.units, scale_object=self._scale_object)
    #     elif isinstance(other, ParametersArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             res = np.empty(self.shape, dtype=object)
    #             for idx in np.ndindex(self.shape):
    #                 res[idx] = self[idx] * other[idx]
    #             item = res[idx]
    #             return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                    units=item.units, scale_object=self._scale_object)
    #         else:
    #             return self * other
    #     else:
    #         return self * other
    #
    # def __rmul__(self, other):
    #     return self.__mul__(other)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, (Parameter, ScalingParameter, float, int)):
    #         res = np.empty(self.shape, dtype=object)
    #         for idx in np.ndindex(self.shape):
    #             res[idx] = self[idx] / other
    #         item = res[idx]
    #         return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                units=item.units, scale_object=self._scale_object)
    #     elif isinstance(other, ParametersArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             res = np.empty(self.shape, dtype=object)
    #             for idx in np.ndindex(self.shape):
    #                 res[idx] = self[idx] / other[idx]
    #             item = res[idx]
    #             return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                    units=item.units, scale_object=self._scale_object)
    #         else:
    #             return self / other
    #     else:
    #         return self / other
    #
    # def __rtruediv__(self, other):
    #     if isinstance(other, (Parameter, ScalingParameter, float, int)):
    #         res = np.empty(self.shape, dtype=object)
    #         for idx in np.ndindex(self.shape):
    #             res[idx] = other / self[idx]
    #         item = res[idx]
    #         return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                units=item.units, scale_object=self._scale_object)
    #     elif isinstance(other, ParametersArray):
    #         if other.shape == self.shape:  # Does not do broadcast
    #             res = np.empty(self.shape, dtype=object)
    #             for idx in np.ndindex(self.shape):
    #                 res[idx] = other / self[idx]
    #             item = res[idx]
    #             return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
    #                                    units=item.units, scale_object=self._scale_object)
    #         else:
    #             return other / self
    #     else:
    #         return other / self
    #
