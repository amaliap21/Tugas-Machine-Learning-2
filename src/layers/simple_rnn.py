from .layer import RNN

class SimpleRNN(RNN):
    def __init__(self, units, activation="tanh", return_sequences=False, go_backwards=False):
        super().__init__(return_sequences=False, go_backwards=False)

        self.key = "simple_rnn"