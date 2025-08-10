
"""

    Variable definition module
    ==========================

    Variables are objects with a name, a symbol and units. They represent any formal quantity in the models.
    
    Description of the classes
    --------------------------

    * :class:`Variable`: Abstract base class to define variables in the models.
    * :class:`VariablesArray`: Base class to define array of variables in the models.

"""

from abc import ABC
from sympy import Symbol
import numpy as np


class Variable(ABC):
    """Abstract base class to define variable object in the models.

    Parameters
    ----------
    name: str
        Name of the variable.
    symbol: ~sympy.core.symbol.Symbol
        A `Sympy`_ symbol to represent the variable in symbolic expressions.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        Latex string representing the variable.
        Empty by default.
    dynamical: bool, optional
        Whether the variable can vary over time.
        Default to `False`.

    Attributes
    ----------
    name: str
        Name of the variable.
    symbol: ~sympy.core.symbol.Symbol
        Symbol of the variable.
    units: str
        The units of the variable specified as atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    latex: str
        Latex string representing the variable.

    .. _Sympy: https://www.sympy.org/
    """

    def __init__(self, name, symbol, units=None, latex=None, dynamical=False):

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

        self._dynamical = dynamical

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ')'

    def __repr__(self):
        return self.__str__()

    @property
    def dynamical(self):
        """bool: Whether the variable can vary over time."""
        return self._dynamical


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
    dynamical: bool, optional
        Whether the variables are varying over time. Default to `False`.

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, values, name, symbol, units="", latex=None, dynamical=False):

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
        arr._units = units
        arr._dynamical = dynamical

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
        self._units = getattr(arr, '_units', "")
        self._dynamical = getattr(arr, '_dynamical', False)

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
    def units(self):
        """str: The units of the dimensional values."""
        return self._units

    @property
    def dynamical(self):
        """bool: Whether the variables can vary over time."""
        return self._dynamical

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
