from .layer import Layer

class Dense(Layer):
    def __init__(self, units, activation=None):
        self.key = "layer"