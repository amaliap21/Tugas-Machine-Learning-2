from .layer import Layer

class Dropout(Layer):
    def __init__(self):
        self.key = "dropout"

    def forward(self, x):
        return x #dropu out layer only for training
    
    def load_keras_weights(self, weights):
        print("Dropout has no trainable weights — skipping")