
import os
import unittest
import numpy as np
from abc import ABC, abstractmethod

path = os.path.abspath('./')
base = os.path.basename(path)
if base == 'model_test':
    fold = ""
else:
    fold = 'model_test/'

real_eps = np.finfo(np.float64).eps


class TestQgsBase(ABC, unittest.TestCase):
    reference = list()
    qgs_values = list()
    layercake_values = list()
    folder = fold

    def load_ref_from_file(self):
        self.reference.clear()
        f = open(self.folder + self.filename, 'r')
        buf = f.readlines()

        for l in buf:
            self.reference.append(l[:-1])

        f.close()

    def save_qgs(self, s):
        self.qgs_values.append(s)

    def save_layercake(self, s):
        self.layercake_values.append(s)

    def check_lists_flt(self):
        self.qgs_outputs()
        self.layercake_outputs()
        for v, r in zip(list(reversed(sorted(self.qgs_values))), list(reversed(sorted(self.layercake_values)))):
            self.assertTrue(self.match_flt(v, r), msg=v + ' != ' + r + ' !!!')

    def check_lists(self, cmax=1):
        self.qgs_outputs()
        self.layercake_outputs()
        for v, r in zip(list(reversed(sorted(self.qgs_values))), list(reversed(sorted(self.layercake_values)))):
            self.assertTrue(self.match_str(v, r, cmax), msg=v + ' != ' + r + ' !!!')

    @abstractmethod
    def qgs_outputs(self):
        pass

    @abstractmethod
    def layercake_outputs(self):
        pass

    @staticmethod
    def match_flt(s1, s2, eps=real_eps):

        s1p = s1.split('=')
        s2p = s2.split('=')

        v1 = float(s1p[1])
        v2 = float(s2p[1])

        return abs(v1 - v2) < eps

    @staticmethod
    def match_str(s1, s2, cmax=1):

        c = 0

        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                c += 1

            if c > cmax:
                return False

        return True
