from .layer import Layer

class Dropout(Layer):
    def __init__(self, rate):
        self.key = "layer"