import numpy as np
from .layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="valid"):
        self.key = "maxpool"
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.padding = padding

    def forward(self, input):
        batch_size, h, w, c = input.shape
        kh, kw = self.pool_size
        sh, sw = self.strides
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        output = np.zeros((batch_size, out_h, out_w, c))

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw
                    output[b, i, j, :] = np.max(input[b, h_start:h_end, w_start:w_end, :], axis=(0, 1))

        return output

    def load_keras_weights(self, weights):
        print("Pooling has no trainable weights — skipping")

class AveragePooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="valid"):
        self.key = "avgpool"
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.padding = padding

    def forward(self, input):
        batch_size, h, w, c = input.shape
        kh, kw = self.pool_size
        sh, sw = self.strides
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        output = np.zeros((batch_size, out_h, out_w, c))

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw
                    output[b, i, j, :] = np.mean(input[b, h_start:h_end, w_start:w_end, :], axis=(0, 1))

        return output

    def load_keras_weights(self, weights):
        print("Pooling has no trainable weights — skipping")
