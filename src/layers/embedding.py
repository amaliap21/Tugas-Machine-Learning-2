from .layer import Layer
from tensorflow.keras.layers import Embedding

class EmbeddingWrapper(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        self.embedding = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.key = 'embedding'
        self.built = False 

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.built = True

    def forward(self, x):
        return self.embedding(x)
    
    def load_keras_weights(self, weights):
        if not self.built:
            dummy_shape = (1, 1)
            self.build(dummy_shape)
        self.embedding.set_weights(weights)
    