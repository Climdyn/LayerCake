
"""

    Classes for formatting symbolic equations output in Fortran
    ===========================================================

    Defines classes to format tendencies and Jacobian symbolic equations output in Fortran.

    Description of the classes
    --------------------------

    * :class:`FortranEquationFormatter`: Class for symbolic equations formatting in Fortran.
    * :class:`FortranJacobianEquationFormatter`: Class for symbolic Jacobian equations formatting in Fortran.

"""

from layercake.formatters.base import EquationFormatter, JacobianEquationFormatter


class FortranEquationFormatter(EquationFormatter):
    """Class for symbolic equations formatting in Fortran.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.

    index_offset: int
        Number that accesses the first element in an array. In Fortran the base index is 1.
    """

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({'conjugate': 'CONJG', })
        self.index_offset = 1

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Fortran language."""
        return '('

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Fortran language."""
        return ')'


class FortranJacobianEquationFormatter(JacobianEquationFormatter):
    """Class for symbolic Jacobian equations formatting in Fortran.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.

    index_offset: int
        Number that accesses the first element in an array. In Fortran the base index is 1.
    """

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({'conjugate': 'CONJG', })
        self.index_offset = 1

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Fortran language."""
        return '('

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Fortran language."""
        return ')'
