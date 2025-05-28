from .layer import Layer
from tensorflow.keras.layers import TextVectorization


class TextVectorizationWrapper(Layer):
    def __init__(self, TextVectorizer):
        self.vectorizer = TextVectorizer

    def forward(self, x):
        return self.vectorizer(x)

    def load_keras_weights(self, weights):
        print("TextVectorization has no trainable weights — skipping")