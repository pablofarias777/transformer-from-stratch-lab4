import numpy as np


class FeedForward:

    def __init__(self, d_model=512, d_ff=2048):

        # inicialização dos pesos (He initialization)
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)

        # bias
        self.b1 = np.zeros((1, d_ff))
        self.b2 = np.zeros((1, d_model))


    def forward(self, x):

        hidden = x @ self.W1 + self.b1

        # ReLU
        hidden = np.maximum(0, hidden)

        output = hidden @ self.W2 + self.b2

        return output