import numpy as np

class FeedForward:

    def __init__(self, d_model=64, d_ff=256):

        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x):

        x = np.maximum(0, x @ self.W1)

        return x @ self.W2