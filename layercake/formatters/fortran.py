
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
    split_lines: bool, optional
        Split the lines if needed, using the `splitter` static function.
        Default to `True`.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.
    index_offset: int
        Number that accesses the first element in an array. In Fortran the base index is 1.
    split_lines: bool
        If `True`, split the lines if needed, using the `splitter` static function.
    """

    def __init__(self, lang_translation=None, split_lines=True):
        EquationFormatter.__init__(self, lang_translation=lang_translation, split_lines=split_lines)
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

    @staticmethod
    def splitter(line, **kwargs):
        return fortran_line_splitter(line, **kwargs)


class FortranJacobianEquationFormatter(JacobianEquationFormatter):
    """Class for symbolic Jacobian equations formatting in Fortran.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.
    split_lines: bool, optional
        Split the lines if needed, using the `splitter` static function.
        Default to `True`.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Fortran language.
    index_offset: int
        Number that accesses the first element in an array. In Fortran the base index is 1.
    split_lines: bool
        If `True`, split the lines if needed, using the `splitter` static function.
    """

    def __init__(self, lang_translation=None, split_lines=True):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation, split_lines=split_lines)
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

    @staticmethod
    def splitter(line, **kwargs):
        return fortran_line_splitter(line, **kwargs)

def fortran_line_splitter(line, line_length=80):
    """Function to split a line of Fortran code that is too long over multiple lines.

    Parameters
    ----------
    line: str
        Line to be split.
    line_length: int, optional
        Length of the resulting split line.
        Default is 80.

    Returns
    -------
    str
       The split line.
    """

    line_out = list()
    line_chunks = [line[i:i+line_length] for i in range(0, len(line), line_length)]
    if len(line_chunks) == 1:
        return line_chunks[0]
    line_out.append(f'{line_chunks[0]}&')
    for chunk in line_chunks[1:-1]:
        line_out.append(f'\t&{chunk}&')
    line_out.append(f'\t{line_chunks[-1]}')
    return "\n".join(line_out)