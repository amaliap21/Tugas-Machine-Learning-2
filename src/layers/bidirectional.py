from .layer import Layer
from .rnn import RNN
from copy import deepcopy
import numpy as np

class Bidirectional(Layer):
    def __init__(self, layer, merge_mode="concat"):
        if not isinstance(layer, RNN):
            raise ValueError(f"Layer must be an instance of RNN, got {type(layer)}")
        
        if merge_mode not in ["concat", "sum", "mul", "ave"]:
            raise ValueError(f"Invalid merge mode: {merge_mode}")
        
        self.key = "bidirectional"
        self.merge_mode = merge_mode

        self.forward_layer = layer
        self.backward_layer = deepcopy(self.forward_layer)
        self.backward_layer.go_backwards = True

    def forward(self, x):
        y_forward = self.forward_layer.forward(x)
        y_backward = self.backward_layer.forward(x)

        if self.forward_layer.return_sequences:
            y_backward = np.flip(y_backward, axis=1)

        if self.merge_mode == "concat":
            return np.concatenate([y_forward, y_backward], axis=-1)
        elif self.merge_mode == "sum":
            return y_forward + y_backward
        elif self.merge_mode == "mul":
            return y_forward * y_backward
        elif self.merge_mode == "ave":
            return (y_forward + y_backward) / 2
        
    def load_keras_weights(self, weights):
        num_weights_per_layer = 3 
        
        forward_weights = weights[:num_weights_per_layer]        
        backward_weights = weights[num_weights_per_layer:]
        
        self.forward_layer.load_keras_weights(forward_weights)
        self.backward_layer.load_keras_weights(backward_weights)