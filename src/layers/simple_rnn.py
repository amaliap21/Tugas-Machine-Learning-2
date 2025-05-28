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

    def forward(self, x):
        """
        note to self:
        batch_size = ammount of sentences
        time_steps = ammount of tokens to represent a sentence
        input_dim = ammount of dimensions used to represent a token
        """
        batch_size, time_steps, input_dim = x.shape
        units = self.kernel.shape[1] # ammount of neuron at end
        h_t = np.zeros((batch_size, units))
        if self.return_sequences:
            outputs = np.zeros((batch_size, time_steps, units))
        time_indices = range(time_steps)
        if self.go_backwards:
            time_indices = reversed(time_indices)
        for t in time_indices:
            x_t = x[:, t, :] # one word of each timestep
            h_t = self.activation(
                np.dot(x_t, self.kernel) + 
                np.dot(h_t, self.recurrent_kernel) + 
                self.bias
            )
            if self.return_sequences:
                outputs[:, t, :] = h_t

        return outputs if self.return_sequences else h_t

    def load_keras_weights(self, weights):
        if len(weights) == 3:
            self.kernel, self.recurrent_kernel, self.bias = weights
            print("Weight successfuly loaded")
        else:
            print(f"Expected 3 weights (kernel, recurrent_kernel, bias), but got {len(weights)}. Cannot load weights.")