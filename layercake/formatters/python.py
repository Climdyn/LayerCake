
"""

    Classes for formatting symbolic equations output in Python
    ==========================================================

    Defines classes to format tendencies and Jacobian symbolic equations output in Python.

    Description of the classes
    --------------------------

    * :class:`PythonEquationFormatter`: Class for symbolic equations formatting in Python.
    * :class:`PythonJacobianEquationFormatter`: Class for symbolic Jacobian equations formatting in Python.

"""

from layercake.formatters.base import EquationFormatter, JacobianEquationFormatter


class PythonEquationFormatter(EquationFormatter):
    """Class for symbolic equations formatting in Python.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Python language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Python language.
    index_offset: int
        Number that accesses the first element in an array. In Python the index base is 0.
    """

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
                                         'sqrt': 'np.sqrt',  # can also be 'math.sqrt'
                                         'lambda': 'lmda',  # remove conflict for lambda function in python
                                     })

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Python language."""
        return '['

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Python language."""
        return ']'


class PythonJacobianEquationFormatter(JacobianEquationFormatter):
    """Class for symbolic Jacobian equations formatting in Python.

    Parameters
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Python language.

    Attributes
    ----------
    lang_translation: dict(str)
        Language translation mapping dictionary, mapping replacements for converting
        Sympy symbolic output strings to the Python language.
    index_offset: int
        Number that accesses the first element in an array. In Python the index base is 0.
    """

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
            'sqrt': 'np.sqrt',  # can also be 'math.sqrt'
            'lambda': 'lmda',  # Remove conflict for lambda function in python
        })

    @property
    def opening_character(self):
        """str: Character opening the arrays specification index in the Python language."""
        return '['

    @property
    def closing_character(self):
        """str: Character closing the arrays specification index in the Python language."""
        return ']'
