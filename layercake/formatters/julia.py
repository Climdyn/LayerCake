
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
    """

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
                                        '**': '^',
                                        'conjugate': 'conj'
                                     })

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
    """

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
            '**': '^',
            'conjugate': 'conj'
        })

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Julia language."""
        return '['

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Julia language."""
        return ']'
