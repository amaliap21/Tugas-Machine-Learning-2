from .layer import Layer
import numpy as np

class Embedding(Layer):
    def __init__(self, input_dim: int, output_dim: int, mask_zero: bool = False):
        self.key = 'embedding'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.weights = None
        
    def forward(self, x):
        if self.weights is None:
            raise ValueError("Weights not loaded. Call load_keras_weights() first.")
        
        if hasattr(x, 'numpy'):
            x = x.numpy()
        else:
            x = np.array(x)
            
        embedding_matrix = self.weights[0]
        batch_size, sequence_length = x.shape
        embedding_dim = embedding_matrix.shape[1]
        output = np.zeros((batch_size, sequence_length, embedding_dim))
        
        mask = None
        if self.mask_zero:
            mask = (x != 0).astype(np.float32)
        
        for i in range(batch_size):
            for j in range(sequence_length):
                token_id = x[i][j]
                if self.mask_zero and token_id == 0:
                    output[i][j] = np.zeros(embedding_dim)
                else:
                    output[i][j] = embedding_matrix[token_id]
        
        if self.mask_zero:
            return output, mask
        else:
            return output
   
    def load_keras_weights(self, weights):
        self.weights = weights