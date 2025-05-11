from .layer import Layer

class MaxPooling1D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="valid"):
        pass

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="valid"):
        pass

class AveragePooling1D(Layer):
    def __init__(self, pool_size, strides=None, padding="valid"):
        pass

class AveragePooling2D(Layer):
    def __init__(self, pool_size, strides=None, padding="valid"):
        pass

class GlobalAveragePooling1D(Layer):
    def __init__(self):
        pass

class GlobalMaxPooling1D(Layer):
    def __init__(self):
        pass

class GlobalAveragePooling2D(Layer):
    def __init__(self):
        pass

class GlobalMaxPooling2D(Layer):
    def __init__(self):
        pass