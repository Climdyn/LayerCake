
from sympy import ImmutableSparseNDimArray
from abc import ABC, abstractmethod
from layercake.utils.symbolic_tensor import get_coords_from_index


class EquationFormatter(ABC):

    def __init__(self, lang_translation=None):

        self.lang_translation = dict()

        if lang_translation is not None:
            self.lang_translation.update(lang_translation)

    def __call__(self, tensor, variable='U', tendencies='F'):
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

        for i in range(ndim):
            for code, new_code in self.lang_translation.items():
                equations_list[i] = equations_list[i].replace(code, new_code)

        return equations_list[1:]

    def _format_components(self, s, idx):
        return f'{s}{self.opening_character}{idx}{self.closing_character}'

    @property
    @abstractmethod
    def opening_character(self):
        pass

    @property
    @abstractmethod
    def closing_character(self):
        pass
