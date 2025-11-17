"""
    Fourier Basis definition module
    ===============================

    Classes and functions defining Fourier basis of functions (`Fourier modes`_) on a plane.

    Description of the classes
    --------------------------

    * :class:`PlanarChannelFourierBasis`: Fourier basis defined on a zonally periodic channel, with no-flux boundary conditions in the meridional direction :math:`y`.
    * :class:`PlanarBasinFourierBasis`: Fourier basis defined on a closed basin, with no-flux boundary conditions in both the zonal and meridional direction :math:`x` and :math:`y`.

    .. _Fourier modes: https://en.wikipedia.org/wiki/Fourier_series

"""
import numpy as np

from layercake.basis.base import SymbolicBasis
from layercake.variables.systems import PlanarCartesianCoordinateSystem
from layercake.variables.parameter import Parameter
from sympy import sin, cos, sqrt, pi


class PlanarChannelFourierBasis(SymbolicBasis):
    """Fourier basis defined on a zonally periodic channel on a 2D plane, with no-flux boundary conditions in the meridional
    direction :math:`y`.

    Parameters
    ----------
    parameters: list(~parameter.Parameter)
        List holding the parameters appearing in the equations defining the basis.
    spectral_blocks: ~numpy.ndarray(int)
        Spectral blocks detailing the modes :math:`x`- and :math:`y`-wavenumber.
        Array of shape (`nblocks`, 2), where `nblocks` is the number of spectral blocks.
    length: float or ~parameter.Parameter, optional
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        Default to `None` for the default length of :math:`2 \\pi / n`.

    Attributes
    ----------
    substitutions: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions. The 2-tuples contain first
        a |Sympy|  expression and then the value to substitute.
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.
    parameters: list(~parameter.Parameter)
        Dictionary holding the parameters appearing in the equations defining the basis.
    length: float or ~parameter.Parameter or None
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        `None` for the default length of :math:`2 \\pi / n`.
    """

    def __init__(self, parameters, spectral_blocks, length=None):

        for param in parameters:
            if str(param.symbol) == 'n':
                break
        else:
            raise ValueError("Parameter 'n' (model aspect ratio) should be present in the provided parameters")

        aspect_ratio = float(param)

        if length is None:
            length = 2 * pi / param.symbol
            # length = 2 * np.pi / aspect_ratio

        if isinstance(length, Parameter):
            self.length = length.symbol
        else:
            self.length = length

        coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., pi)))
        # coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., aspect_ratio * length / 2)))
        SymbolicBasis.__init__(self, coordinate_system, parameters)
        self._n = param.symbol
        self.substitutions.append((self._n, aspect_ratio))

        awavenum = channel_wavenumbers(spectral_blocks)

        for wv in awavenum:

            mode_eq = fourier_functions(wv, self._n, self.coordinate_system)

            if mode_eq is not None:
                self.functions.append(mode_eq)

    def set_parameters(self, parameters):
        """Setter for the parameters' dictionary.

        Attributes
        ----------
        parameters: list(~parameter.Parameter)
            List holding the parameters appearing in the equations defining the basis.
        """

        for param in parameters:
            if str(param.symbol) == 'n':
                break
        else:
            raise ValueError("Parameter 'n' (model aspect ratio) should be present in the provided parameters")

        aspect_ratio = float(param)

        if self.length is None:
            length = 2 * pi / param.symbol
        elif isinstance(self.length, Parameter):
            length = self.length.symbol
        else:
            length = self.length

        coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., pi)))
        self.coordinate_system = coordinate_system
        self._n = param.symbol
        self.substitutions = list()
        self.substitutions.append((self._n, aspect_ratio))


class PlanarBasinFourierBasis(SymbolicBasis):
    """Fourier basis defined on a closed basin, with no-flux boundary conditions in both the zonal and meridional
    direction :math:`x` and :math:`y`.

    Parameters
    ----------
    parameters: list(~parameter.Parameter)
        List holding the parameters appearing in the equations defining the basis.
    spectral_blocks: ~numpy.ndarray(int)
        Spectral blocks detailing the modes :math:`x`- and :math:`y`-wavenumber.
        Array of shape (`nblocks`, 2), where `nblocks` is the number of spectral blocks.
    length: float or ~parameter.Parameter, optional
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        Default to `None` for the default length of :math:`2 \\pi / n`.

    Attributes
    ----------
    substitutions: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions. The 2-tuples contain first
        a |Sympy|  expression and then the value to substitute.
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.
    parameters: list(~parameter.Parameter)
        Dictionary holding the parameters appearing in the equations defining the basis.
    length: float or None
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        `None` for the default length of :math:`2 \\pi / n`.

    """

    def __init__(self, parameters, spectral_blocks, length=None):

        for param in parameters:
            if str(param.symbol) == 'n':
                break
        else:
            raise ValueError("Parameter 'n' (model aspect ratio) should be present in the provided parameters")

        aspect_ratio = float(param)

        if length is None:
            length = 2 * pi / param.symbol
            # length = 2 * np.pi / aspect_ratio

        if isinstance(length, Parameter):
            self.length = length.symbol
        else:
            self.length = length

        coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., pi)))
        # coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., aspect_ratio * length / 2)))
        SymbolicBasis.__init__(self, coordinate_system, parameters)
        self._n = param.symbol
        self.substitutions.append((self._n, aspect_ratio))

        owavenum = basin_wavenumbers(spectral_blocks)

        for wv in owavenum:

            mode_eq = fourier_functions(wv, self._n, self.coordinate_system)

            if mode_eq is not None:
                self.functions.append(mode_eq)

    def set_parameters(self, parameters):
        """Setter for the parameters' dictionary.

        Attributes
        ----------
        parameters: list(~parameter.Parameter)
            List holding the parameters appearing in the equations defining the basis.
        """

        for param in parameters:
            if str(param.symbol) == 'n':
                break
        else:
            raise ValueError("Parameter 'n' (model aspect ratio) should be present in the provided parameters")

        aspect_ratio = float(param)

        if self.length is None:
            length = 2 * pi / param.symbol
        elif isinstance(self.length, Parameter):
            length = self.length.symbol
        else:
            length = self.length

        coordinate_system = PlanarCartesianCoordinateSystem(extent=((0., length), (0., pi)))
        self.coordinate_system = coordinate_system
        self._n = param.symbol
        self.substitutions = list()
        self.substitutions.append((self._n, aspect_ratio))


def contiguous_basin_basis(nxmax, nymax, parameters, length=None):
    """Function that returns the basis for contiguous spectral blocks of modes on a closed basin.

    Parameters
    ----------
    nxmax: int
        Maximum x-wavenumber to fill the spectral block up to.
    nymax: int
        Maximum :math:`y`-wavenumber to fill the spectral block up to.
    parameters: list(~parameter.Parameter)
        List holding the parameters appearing in the equations defining the basis.
    length: float or ~parameter.Parameter, optional
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        Default to `None` for the default length of :math:`2 \\pi / n`.

    Returns
    -------
    PlanarBasinFourierBasis
        The closed basin contiguous basis up to the specified spectral truncation.
    """

    spectral_blocks = np.zeros((nxmax * nymax, 2), dtype=int)
    i = 0
    for nx in range(1, nxmax + 1):
        for ny in range(1, nymax+1):
            spectral_blocks[i, 0] = nx
            spectral_blocks[i, 1] = ny
            i += 1

    return PlanarBasinFourierBasis(parameters, spectral_blocks, length)


def contiguous_channel_basis(nxmax, nymax, parameters, length=None):
    """Function that returns the basis for contiguous spectral blocks of modes on a channel.

    Parameters
    ----------
    nxmax: int
        Maximum x-wavenumber to fill the spectral block up to.
    nymax: int
        Maximum :math:`y`-wavenumber to fill the spectral block up to.
    parameters: list(~parameter.Parameter)
        List holding the parameters appearing in the equations defining the basis.
    length: float or ~parameter.Parameter, optional
        Length of the domain along the :math:`x` coordinate: :math:`L_x` .
        Default to `None` for the default length of :math:`2 \\pi / n`.

    Returns
    -------
    ChannelFourierBasis
        The channel contiguous basis up to the specified spectral truncation.
    """

    spectral_blocks = np.zeros((nxmax * nymax, 2), dtype=int)
    i = 0
    for nx in range(1, nxmax + 1):
        for ny in range(1, nymax+1):
            spectral_blocks[i, 0] = nx
            spectral_blocks[i, 1] = ny
            i += 1

    return PlanarChannelFourierBasis(parameters, spectral_blocks, length)


def fourier_functions(wave_number, n, coordinate_system):
    """Function that return Fourier modes expressions:

    * `'A'` for a function of the form :math:`F^A_{P} (x, y) =  \\sqrt{2}\\, \\cos(P y) = \\sqrt{2}\\, \\cos(n_y\\, y)`
    * `'K'` for a function of the form :math:`F^K_{M,P} (x, y) =  2\\cos(M nx)\\, \\sin(P y) = 2\\cos(n_x\\,  n\\, x)\\, \\sin(n_y\\, y)`
    * `'L'` for a function of the form :math:`F^L_{H,P} (x, y) = 2\\sin(H nx)\\, \\sin(P y) = 2\\sin(n_x\\, n \\,x)\\, \\sin(n_y\\, y)`

    Parameters
    ----------
    wave_number: WaveNumber
        The wavenumber and type information of the mode to be returned.
    n: ~sympy.core.symbol.Symbol
        The aspect ratio symbol.
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.

    Returns
    -------
    ~sympy.core.expr.Expr
        Symbolic expression of the mode.

    """
    mode_eq = None
    x = coordinate_system.coordinates_symbol['x']
    y = coordinate_system.coordinates_symbol['y']
    if wave_number.type == 'A':
        mode_eq = sqrt(2) * cos(wave_number.ny * y)
    elif wave_number.type == 'K':
        mode_eq = 2 * cos(wave_number.nx * n * x) * sin(wave_number.ny * y)
    elif wave_number.type == 'L':
        mode_eq = 2 * sin(wave_number.nx * n * x) * sin(wave_number.ny * y)

    return mode_eq


class WaveNumber(object):
    """Class to define model base functions wavenumber. The basis function available are:

    * `'A'` for a function of the form :math:`F^A_{P} (x, y) =  \\sqrt{2}\\, \\cos(P y) = \\sqrt{2}\\, \\cos(n_y\\, y)`
    * `'K'` for a function of the form :math:`F^K_{M,P} (x, y) =  2\\cos(M nx)\\, \\sin(P y) = 2\\cos(n_x\\,  n\\, x)\\, \\sin(n_y\\, y)`
    * `'L'` for a function of the form :math:`F^L_{H,P} (x, y) = 2\\sin(H nx)\\, \\sin(P y) = 2\\sin(n_x\\, n \\,x)\\, \\sin(n_y\\, y)`

    where :math:`x` and :math:`y` are the nondimensional model's domain coordinates.

    Parameters
    ----------
    function_type: str
        One character string to define the type of basis function. It can be `'A'`, `'K'` or `'L'`.
    P: int
        The :math:`y` wavenumber integer.
    M: int
        The :math:`x` wavenumber integer.
    H: int
        The :math:`x` wavenumber integer.
    nx: float
        The :math:`x` wavenumber.
    ny: float
        The :math:`y` wavenumber.

    Attributes
    ----------
    type: str
        One character string to define the type of basis function. It can be `'A'`, `'K'` or `'L'`.
    P: int
        The :math:`y` wavenumber integer.
    M: int
        The :math:`x` wavenumber integer.
    H: int
        The :math:`x` wavenumber integer.
    nx: float
        The :math:`x` wavenumber.
    ny: float
        The :math:`y` wavenumber.

    """

    def __init__(self, function_type, P, M, H, nx, ny):
        self.type = function_type
        self.P = P
        self.M = M
        self.H = H
        self.nx = nx
        self.ny = ny

    def __repr__(self):
        return "type = {}, P = {}, M= {},H={}, nx= {}, ny={}".format(self.type, self.P, self.M, self.H, self.nx, self.ny)


def channel_wavenumbers(spectral_blocks):
    """Functions that returns the :class:`WaveNumber` objects corresponding to a given list of spectral blocks for a
    channel-like spatial domain.

    Parameters
    ----------
    spectral_blocks: ~numpy.ndarray(int)
        Spectral blocks detailing the modes :math:`x`- and :math:`y`-wavenumber.
        Array of shape (nblocks, 2), where `nblocks` is the number of spectral blocks.

    Returns
    -------
    ~numpy.ndarray(WaveNumber)
        The array of wavenumber objects corresponding to the given spectral blocks.
    """
    # initialization of the variables
    wave_numbers = list()

    # Atmospheric wavenumbers definition

    for i in range(spectral_blocks.shape[0]):  # function type is limited to AKL for the moment: atmosphere is a channel

        if spectral_blocks[i, 0] == 1:
            wave_numbers.append(WaveNumber('A', spectral_blocks[i, 1], 0, 0, 0, spectral_blocks[i, 1]))
            wave_numbers.append(WaveNumber('K', spectral_blocks[i, 1], spectral_blocks[i, 0], 0, spectral_blocks[i, 0], spectral_blocks[i, 1]))
            wave_numbers.append(WaveNumber('L', spectral_blocks[i, 1], 0, spectral_blocks[i, 0], spectral_blocks[i, 0], spectral_blocks[i, 1]))
        else:
            wave_numbers.append(WaveNumber('K', spectral_blocks[i, 1], spectral_blocks[i, 0], 0, spectral_blocks[i, 0], spectral_blocks[i, 1]))
            wave_numbers.append(WaveNumber('L', spectral_blocks[i, 1], 0, spectral_blocks[i, 0], spectral_blocks[i, 0], spectral_blocks[i, 1]))

    return np.array(wave_numbers)


def basin_wavenumbers(spectral_blocks):
    """Functions that returns the :class:`WaveNumber` objects corresponding to a given list of spectral blocks for a
    closed basin spatial domain.

    Parameters
    ----------
    spectral_blocks: ~numpy.ndarray(int)
        Spectral blocks detailing the modes :math:`x`- and :math:`y`-wavenumber.
        Array of shape (`nblocks`, 2), where `nblocks` is the number of spectral blocks.

    Returns
    -------
    ~numpy.ndarray(WaveNumber)
        The array of wavenumber objects corresponding to the given spectral blocks.
    """

    # initialization of the variables
    wave_numbers = list()

    # Oceanic wavenumbers definition

    for i in range(spectral_blocks.shape[0]):  # function type is limited to L for the moment: ocean is a closed basin
        wave_numbers.append(WaveNumber('L', spectral_blocks[i, 1], 0, spectral_blocks[i, 0], spectral_blocks[i, 0] / 2., spectral_blocks[i, 1]))

    return np.array(wave_numbers)
