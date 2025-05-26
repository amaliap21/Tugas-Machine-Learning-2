from .layer import Layer

class RNN(Layer):
    def __init__(self, return_sequences=False, go_backwards=False):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.key = "rnn"