from .layer import Layer

class SimpleRNN(Layer):
    def __init__(self, units, activation="tanh"):
        self.key = "layer"