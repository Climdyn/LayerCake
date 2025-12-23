
"""

    Classes for formatting symbolic equations output in Julia
    ==========================================================

    Defines classes to format tendencies and Jacobian symbolic equations output in Julia.

    Description of the classes
    --------------------------

    * :class:`JuliaEquationFormatter`: Class for symbolic equations formatting in Julia.
    * :class:`JuliaJacobianEquationFormatter`: Class for symbolic Jacobian equations formatting in Julia.

"""

from layercake.formatters.base import EquationFormatter, JacobianEquationFormatter


class JuliaEquationFormatter(EquationFormatter):
    """Class for symbolic equations formatting in Julia.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Julia language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Julia language.
    index_offset: int
        Number that accesses the first element in an array. In Julia the index base is 1.
    """

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
                                        '**': '^',
                                        'conjugate': 'conj'
                                     })
        
        self.index_offset = 1

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Julia language."""
        return '['

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Julia language."""
        return ']'


class JuliaJacobianEquationFormatter(JacobianEquationFormatter):
    """Class for symbolic Jacobian equations formatting in Julia.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Julia language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Julia language.
    index_offset: int
        Number that accesses the first element in an array. In Julia the index base is 1.
    """

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
            '**': '^',
            'conjugate': 'conj'
        })

        self.index_offset = 1

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Julia language."""
        return '['

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Julia language."""
        return ']'
