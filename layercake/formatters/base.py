
"""

    Base classes for formatting symbolic equations output
    =====================================================

    Defines base classes to format tendencies and Jacobian symbolic equations output.

    Description of the classes
    --------------------------

    * :class:`EquationFormatter`: Base class for symbolic equations formatting.
    * :class:`JacobianEquationFormatter`: Base class for symbolic Jacobian equations formatting.

"""

from sympy import ImmutableSparseNDimArray
from abc import ABC, abstractmethod
from layercake.utils.symbolic_tensor import get_coords_from_index


class EquationFormatter(ABC):
    """Base class for symbolic equations formatting.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the target language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the target language.

    index_offset: int
        Number that accesses the first element in an array. Defulats to 0.
    """

    def __init__(self, lang_translation=None):

        self.lang_translation = dict()
        self.index_offset = 0

        if lang_translation is not None:
            self.lang_translation.update(lang_translation)

    def __call__(self, tensor, variable='U', tendencies='F'):
        """Convert a model symbolic tendencies terms tensor to a list of symbolic equations in
        string format.

        Parameters
        ----------
        tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
            Symbolic tendencies terms tensor to convert.
        variable: str
            Name of the state variable to use for the output equations strings.
            Default to `'U'`.
        tendencies: str
            Name of the tendencies variable to use for the output equations strings.
            Default to `'F`.
        """
        if not isinstance(tensor, ImmutableSparseNDimArray):
            raise ValueError('Only symbolic tensor can be converted to symbolic equations.')

        ndim = tensor.shape[0]
        shape_len = len(tensor.shape)
        equations_list = list()
        for i in range(ndim):
            equations_list.append(f'{self._format_components(tendencies, i)} = ')
        for i in range(1, ndim):
            for n, val in tensor[i]._args[0].items():
                coords = get_coords_from_index(n, ndim, shape_len-1)
                new_term = f'{val} '
                if new_term[0] != '-':
                    new_term = '+' + new_term
                for c in coords:
                    if c != 0:
                        new_term += f'* {self._format_components(variable, c)} '
                equations_list[i] += new_term

        for i in range(1, ndim):
            for code, new_code in self.lang_translation.items():
                equations_list[i] = equations_list[i].replace(code, new_code)

        return equations_list[1:]

    def _format_components(self, s, idx):
        # Index offset included to allow for differnet language index bases
        return f'{s}{self.opening_character}{idx + self.index_offset - 1}{self.closing_character}'

    @property
    @abstractmethod
    def opening_character(self):
        """str: Character opening the arrays specification index in the target language.
        Must be defined in the subclasses."""
        pass

    @property
    @abstractmethod
    def closing_character(self):
        """str: Character closing the arrays specification index in the target language.
        Must be defined in the subclasses."""
        pass


class JacobianEquationFormatter(EquationFormatter):
    """Base class for symbolic Jacobian equations formatting.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the target language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the target language.
    """

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)

    def __call__(self, tensor, variable='U', tendencies='J'):
        """Convert a model Jacobian symbolic tendencies terms tensor to a list of symbolic equations in
        string format.

        Parameters
        ----------
        tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
            Jacobian symbolic tendencies terms tensor to convert.
        variable: str
            Name of the state variable to use for the output equations strings.
            Default to `'U'`.
        tendencies: str
            Name of the tendencies variable to use for the output equations strings.
            Default to `'F`.
        """
        if not isinstance(tensor, ImmutableSparseNDimArray):
            raise ValueError('Only symbolic tensor can be converted to symbolic equations.')

        ndim = tensor.shape[0]
        shape_len = len(tensor.shape)
        equations_matrix = list()
        for i in range(ndim):
            equations_matrix.append(list())
            for j in range(ndim):
                equations_matrix[-1].append(None)
        for i in range(1, ndim):
            for n, val in tensor[i]._args[0].items():
                coords = get_coords_from_index(n, ndim, shape_len-1)
                j = coords[0]
                if equations_matrix[i][j] is None:
                    equations_matrix[i][j] = f'{tendencies}{self.opening_character}{i + self.index_offset - 1},{j + self.index_offset - 1}{self.closing_character} = '
                new_term = f'{val} '
                if new_term[0] != '-':
                    new_term = '+' + new_term
                for c in coords[1:]:
                    if c != 0:
                        new_term += f'* {self._format_components(variable, c)} '
                equations_matrix[i][j] += new_term

        equations_list = list()
        for i in range(1, ndim):
            for j in range(1, ndim):
                if equations_matrix[i][j] is not None:
                    eq = equations_matrix[i][j]
                    for code, new_code in self.lang_translation.items():
                        eq = eq.replace(code, new_code)

                    equations_list.append(eq)

        return equations_list
