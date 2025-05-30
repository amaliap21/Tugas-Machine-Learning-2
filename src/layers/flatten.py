import numpy as np
from .layer import Layer

class Flatten(Layer):
    def __init__(self):
        self.key = "flatten"

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def load_keras_weights(self, weights):
        print("Flatten has no trainable weights — skipping")