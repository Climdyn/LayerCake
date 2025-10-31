
from layercake.formatters.base import EquationFormatter, JacobianEquationFormatter


class FortranEquationFormatter(EquationFormatter):

    def __init__(self, lang_translation=None):
        EquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({'conjugate': 'CONJG', })

    @property
    def opening_character(self):
        return '('

    @property
    def closing_character(self):
        return ')'


class FortranJacobianEquationFormatter(JacobianEquationFormatter):

    def __init__(self, lang_translation=None):
        JacobianEquationFormatter.__init__(self, lang_translation=lang_translation)
        self.lang_translation.update({'conjugate': 'CONJG', })

    @property
    def opening_character(self):
        return '('

    @property
    def closing_character(self):
        return ')'
