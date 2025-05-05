

class Cake(object):

    def __int__(self):

        self.layers = list()

    def add_layer(self, layer):
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
