
from layercake.formatters.base import EquationFormatter, JacobianEquationFormatter


class PythonEquationFormatter(EquationFormatter):

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
                                         'sqrt': 'np.sqrt',  # can also be 'math.sqrt'
                                         'lambda': 'lmda',  # remove conflict for lambda function in python
                                     })

    @property
    def opening_character(self):
        return '['

    @property
    def closing_character(self):
        return ']'


class PythonJacobianEquationFormatter(JacobianEquationFormatter):

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({
            'sqrt': 'np.sqrt',  # can also be 'math.sqrt'
            'lambda': 'lmda',  # Remove conflict for lambda function in python
        })

    @property
    def opening_character(self):
        return '['

    @property
    def closing_character(self):
        return ']'
