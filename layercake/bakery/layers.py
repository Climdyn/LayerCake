
import numpy as np
import sparse as sp


class Layer(object):

    def __init__(self):

        self.equations = list()
        self.tensor = None
        self._cake = None
        self._order = 0

    def add_equation(self, equation):
        self.equations.append(equation)
        equation._layer = self

    @property
    def fields(self):
        fields_list = list()
        for eq in self.equations:
            fields_list.append(eq.field)
        return fields_list

    @property
    def ndim(self):
        dim = 0
        for field in self.fields:
            dim += field.state.__len__()
        return dim

    @property
    def number_of_equations(self):
        return self.equations.__len__()

    @property
    def maximum_rank(self):
        max_rank = 0
        for eq in self.equations:
            max_rank = max(max_rank, eq.maximum_rank)
        return max_rank

    def compute_inner_products(self, numerical=True):
        for field, eq in zip(self.fields, self.equations):
            eq.lhs_term.compute_inner_products(field.basis, numerical=numerical)
            for term in eq.terms:
                term.compute_inner_products(field.basis, numerical=numerical)

    def compute_tensor(self, numerical=True, compute_inner_products=False):

        if compute_inner_products:
            self.compute_inner_products(numerical=numerical)

        if numerical:
            if self._cake is not None:
                # self._order = ...
                shape = tuple([self._cake.ndim + 1] + [self.ndim + 1] * (self.maximum_rank - 1))
            else:
                self._order = 0
                shape = tuple([self.ndim] + [self.ndim + 1] * (self.maximum_rank - 1))

            self.tensor = sp.zeros(shape, dtype=np.float64, format='dok')
            lhs_mat = np.zeros((self.ndim, self.ndim))
            orderf = 0
            order = 1
            for field, eq in zip(self.fields, self.equations):
                ndim = field.state.__len__()
                lhs_mat[orderf:orderf+ndim, orderf:orderf+ndim] = eq.lhs_term.inner_products.todense()
                for term in eq.terms:
                    slices = [slice(orderf, orderf+ndim)] + [slice(order, order+ndim) for _ in range(term.rank-1)]
                    zeros = [0 for _ in range(term.rank, len(self.tensor.shape))]
                    args = tuple(slices+zeros)
                    self.tensor[args] = self.tensor[args] + term.inner_products.todense()
                order += ndim
                orderf += ndim
            self.tensor = sp.DOK(np.tensordot(np.linalg.inv(lhs_mat), self.tensor.to_coo(), 1))

        else:
            pass

