import numpy as np

class Layer():
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")
    
    def load_keras_weights(self, weights):
        raise NotImplementedError("Load Keras weights method not implemented")
    
    @staticmethod
    def _get_activation(name):
        if name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == "tanh":
            return np.tanh
        elif name == "softmax":
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e_x / np.sum(e_x, axis=-1, keepdims=True)
            return softmax
        elif name is None:
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation function '{name}'")