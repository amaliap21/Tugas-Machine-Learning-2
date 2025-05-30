from .layer import Layer
import numpy as np

class Embedding(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        self.key = 'embedding'
        self.weights = None

    def forward(self, x):
        if self.weights is None:
            raise ValueError("Weights not loaded. Call load_keras_weights() first.")
        embedding_matrix = self.weights[0]
        batch_size, sequence_length = x.shape
        embedding_dim = embedding_matrix.shape[1]
        output = np.zeros((batch_size, sequence_length, embedding_dim))
        for i in range(batch_size):
            for j in range(sequence_length):
                token_id = x[i][j]
                output[i][j] = embedding_matrix[token_id] # a word is converted to a single vector (row)
        return output
    
    def load_keras_weights(self, weights):
        self.weights= weights
