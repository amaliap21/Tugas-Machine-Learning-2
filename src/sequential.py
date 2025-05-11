from layers.layer import Layer

class Sequential:
    def __init__(self, layers=None):
        self.layers = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Layer must be an instance of Layer class.")
        self.layers.append(layer)