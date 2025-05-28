import numpy as np
from layer import Layer

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', groups=1, activation=None):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding.lower()
        self.groups = groups
        self.activation = self._get_activation(activation)
        self.weights = None
        self.bias = None

    def load_keras_weights(self, weights):
        self.weights = weights[0]  
        self.bias = weights[1]     

    def _pad_input(self, input):
        if self.padding == 'valid':
            return input
        elif self.padding == 'same':
            in_h, in_w = input.shape[1:3]
            pad_h = max((in_h - 1) * self.strides[0] + self.kernel_size[0] - in_h, 0)
            pad_w = max((in_w - 1) * self.strides[1] + self.kernel_size[1] - in_w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            return np.pad(input, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            raise ValueError(f"Tipe padding tidak valid: {self.padding}")

    def forward(self, input):
        batch_size = input.shape[0]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        input_padded = self._pad_input(input)
        padded_h, padded_w = input_padded.shape[1:3]

        out_h = (padded_h - kernel_h) // stride_h + 1
        out_w = (padded_w - kernel_w) // stride_w + 1

        output = np.zeros((batch_size, out_h, out_w, self.filters))

        for b in range(batch_size):
            for f in range(self.filters):
                kernel = self.weights[:, :, :, f]
                bias = self.bias[f]
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        h_end = h_start + kernel_h
                        w_start = j * stride_w
                        w_end = w_start + kernel_w

                        patch = input_padded[b, h_start:h_end, w_start:w_end, :]
                        output[b, i, j, f] = np.sum(patch * kernel) + bias

        self.key = "layer"
        return self.activation(output)
