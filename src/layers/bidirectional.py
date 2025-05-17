from .layer import Layer

class Bidirectional(Layer):
    def __init__(self, layer, merge_mode="concat", weights=None, backward_layer=None):
        self.key = "layer"