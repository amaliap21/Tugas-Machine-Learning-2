from .rnn import RNN
import numpy as np

class SimpleRNN(RNN):
    def __init__(self, units, activation="tanh", return_sequences=False, go_backwards=False):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.key = "simple_rnn"
        self.kernel = None # U
        self.recurrent_kernel = None # W
        self.bias = None # b
        self.activation = self._get_activation(activation)
        
    def forward(self, x, mask=None):
        batch_size, time_steps, input_dim = x.shape
        units = self.kernel.shape[1]
        h_t = np.zeros((batch_size, units))

        if self.return_sequences:
            outputs = np.zeros((batch_size, time_steps, units))

        time_indices = list(range(time_steps))
        if self.go_backwards:
            time_indices = list(reversed(time_indices))

        for step_idx, t in enumerate(time_indices):
            x_t = x[:, t, :]

            h_t_new = self.activation(
                np.dot(x_t, self.kernel) +
                np.dot(h_t, self.recurrent_kernel) +
                self.bias
            )

            if mask is not None:
                mask_t = mask[:, t]
                mask_t = mask_t[:, np.newaxis] 
                h_t = mask_t * h_t_new + (1 - mask_t) * h_t
            else:
                h_t = h_t_new

            if self.return_sequences:
                if self.go_backwards:
                    outputs[:, time_steps - 1 - step_idx, :] = h_t
                else:
                    outputs[:, t, :] = h_t

        if self.return_sequences:
            return outputs
        return h_t

    def load_keras_weights(self, weights):
        if len(weights) == 3:
            self.kernel, self.recurrent_kernel, self.bias = weights
            print("Weight successfuly loaded")
        else:
            print(f"Expected 3 weights (kernel, recurrent_kernel, bias), but got {len(weights)}. Cannot load weights.")