from .layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, units, activation=None):
        self.key = "dense"
        self.units = units
        self.activation = self._get_activation(activation)

        self.kernel = None
        self.bias = None

    def forward(self, x):
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights not initialized. Call load_keras_weights() first.")

        z = np.dot(x, self.kernel) + self.bias
        output = self.activation(z)

        return output

    def load_keras_weights(self, weights):
        self.kernel = weights[0]
        self.bias = weights[1]