
import numpy as np
import sparse as sp
from numba import njit
from layercake.utils.tensor import sparse_mul, jsparse_mul


class Cake(object):

    def __init__(self):

        self.layers = list()

    def add_layer(self, layer):
        layer._cake_order = len(self.layers)
        self.layers.append(layer)
        layer._cake = self

    @property
    def ndim(self):
        dim = 0
        for layer in self.layers:
            dim += layer.ndim
        return dim

    @property
    def number_of_layers(self):
        return self.layers.__len__()

    def compute_tensor(self, numerical=True, compute_inner_products=False):

        for layer in self.layers:
            layer.compute_tensor(numerical, compute_inner_products)

    @property
    def maximum_rank(self):
        max_rank = 0
        for layer in self.layers:
            max_rank = max(max_rank, layer.maximum_rank)
        return max_rank

    @property
    def _layers_first_index(self):
        idx = [1]
        for layer in self.layers[:-1]:
            idx.append(idx[-1] + layer.ndim)
        return idx

    @property
    def _layers_last_index(self):
        idx = list()
        for i, layer in enumerate(self.layers):
            if i == 0:
                idx.append(layer.ndim)
            else:
                idx.append(idx[-1] + layer.ndim + 1)
        return idx

    @property
    def tensor(self):
        if isinstance(self.layers[0].tensor, sp.COO):
            shape = tuple([self.ndim + 1] * self.maximum_rank)
            tensor = sp.zeros(shape, dtype=np.float64, format='dok')
            for i, layer in enumerate(self.layers):
                lmax = layer.maximum_rank
                if i < self.number_of_layers - 1:
                    slices = [slice(self._layers_first_index[i], self._layers_first_index[i + 1])] + [slice(0, None) for _ in range(lmax - 1)]
                    zeros = [0 for _ in range(lmax, len(layer.tensor.shape))]
                else:
                    slices = [slice(self._layers_first_index[i], None)] + [slice(0, None) for _ in range(lmax - 1)]
                    zeros = [0 for _ in range(lmax, len(layer.tensor.shape))]
                args = tuple(slices + zeros)
                tensor[args] = tensor[args] + layer.tensor.todense()[1:]
                tensor = tensor.to_coo()
        else:
            tensor = None

        return tensor

    @property
    def jacobian_tensor(self):
        tensor = self.tensor
        if isinstance(tensor, sp.COO):
            return self._jacobian_from_tensor(tensor)
        else:
            return None

    @staticmethod
    def _jacobian_from_tensor(tensor):
        """Function to compute the Jacobian tensor.

        Parameters
        ----------
        tensor: sparse.COO
            The qgs tensor.

        Returns
        -------
        sparse.COO
            The Jacobian tensor.
        """

        n_perm = len(tensor.shape) - 2

        jacobian_tensor = tensor.copy()

        for i in range(1, n_perm+1):
            jacobian_tensor += tensor.swapaxes(1, i+1)

        return jacobian_tensor

    def compute_tendencies(self):
        if self.tensor is not None:

            if isinstance(self.tensor, sp.COO):
                coo = self.tensor.coords.T
                val = self.tensor.data

                @njit
                def f(t, x):
                    xx = np.concatenate((np.full((1,), 1.), x))
                    xr = sparse_mul(xx, coo, val)
                    return xr[1:]

                jcoo = self.jacobian_tensor.coords.T
                jval = self.jacobian_tensor.data

                @njit
                def Df(t, x):
                    xx = np.concatenate((np.full((1,), 1.), x))
                    mul_jac = jsparse_mul(xx, jcoo, jval)
                    return mul_jac[1:, 1:]

                return f, Df
            else:
                return None, None
        else:
            return None, None
