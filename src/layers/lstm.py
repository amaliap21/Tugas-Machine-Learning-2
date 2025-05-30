from .rnn import RNN
import numpy as np

class LSTM(RNN):
    def __init__(self, units, activation="tanh", recurrent_activation="sigmoid", return_sequences=False, go_backwards=False):
        super().__init__(return_sequences=return_sequences, go_backwards=go_backwards)

        self.key = "lstm"
        self.units = units
        self.activation = self._get_activation(activation)
        self.recurrent_activation = self._get_activation(recurrent_activation)

        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None

    def forward(self, x, mask=None):
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("Weights not initialized. Call load_keras_weights() first.")

        original_ndim = x.ndim
        if original_ndim == 2:
            timesteps, _ = x.shape
            batch_size = 1
            # Reshape x_sequence to (batch_size, timesteps, input_dim)
            x_batched = x[np.newaxis, :, :]
            if mask is not None:
                mask = mask[np.newaxis, :]
        elif original_ndim == 3:
            batch_size, timesteps, _ = x.shape
            x_batched = x
        else:
            raise ValueError("Input sequence must be 2D or 3D.")

        h_t = np.zeros((batch_size, self.units))
        c_t = np.zeros((batch_size, self.units))

        outputs = np.zeros((batch_size, timesteps, self.units))

        W_i = self.kernel[:, :self.units]
        W_f = self.kernel[:, self.units:self.units * 2]
        W_c = self.kernel[:, self.units * 2:self.units * 3]
        W_o = self.kernel[:, self.units * 3:]

        U_i = self.recurrent_kernel[:, :self.units]
        U_f = self.recurrent_kernel[:, self.units:self.units * 2]
        U_c = self.recurrent_kernel[:, self.units * 2:self.units * 3]
        U_o = self.recurrent_kernel[:, self.units * 3:]

        b_i = self.bias[:self.units]
        b_f = self.bias[self.units:self.units * 2]
        b_c = self.bias[self.units * 2:self.units * 3]
        b_o = self.bias[self.units * 3:]  

        time_range = reversed(range(timesteps)) if self.go_backwards else range(timesteps)

        for step_idx, t in enumerate(time_range):
            x_t = x_batched[:, t, :]
            
            if mask is not None:
                mask_t = mask[:, t:t+1]
                
                if np.all(mask_t == 0):
                    if self.return_sequences:
                        if self.go_backwards:
                            outputs[:, timesteps - 1 - step_idx, :] = h_t
                        else:
                            outputs[:, t, :] = h_t
                    continue

            f_gate = self.recurrent_activation(np.dot(x_t, W_f) + np.dot(h_t, U_f) + b_f)
            i_gate = self.recurrent_activation(np.dot(x_t, W_i) + np.dot(h_t, U_i) + b_i)
            c_candidate = self.activation(np.dot(x_t, W_c) + np.dot(h_t, U_c) + b_c)
            o_gate = self.recurrent_activation(np.dot(x_t, W_o) + np.dot(h_t, U_o) + b_o)

            c_t_new = f_gate * c_t + i_gate * c_candidate
            h_t_new = o_gate * self.activation(c_t_new)

            if mask is not None:
                mask_expanded = np.broadcast_to(mask_t, h_t.shape)

                h_t = mask_expanded * h_t_new + (1 - mask_expanded) * h_t
                c_t = mask_expanded * c_t_new + (1 - mask_expanded) * c_t
            else:
                h_t = h_t_new
                c_t = c_t_new

            if self.return_sequences:
                if self.go_backwards:
                    outputs[:, timesteps - 1 - step_idx, :] = h_t
                else:
                    outputs[:, t, :] = h_t

        if self.return_sequences:
            if original_ndim == 2:
                return outputs[0]
            return outputs
        else:
            if original_ndim == 2:
                return h_t[0]
            return h_t

    def load_keras_weights(self, weights):
        self.kernel = weights[0]
        self.recurrent_kernel = weights[1]
        self.bias = weights[2]