from .layer import Layer
from tensorflow.keras.layers import TextVectorization


class TextVectorizationWrapper(Layer):
    def __init__(self, max_tokens:int, output_sequence_length: int, output_mode:str='int'):
        super().__init__()
        self.vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode=output_mode,
            output_sequence_length=output_sequence_length
        )
        self.key = "text_vectorization"
        
    def forward(self, x):
        self.vectorizer.adapt(x)
        return self.vectorizer(x)
    
    def load_keras_weights(self, weights):
        print("TextVectorization has no trainable weights — skipping")