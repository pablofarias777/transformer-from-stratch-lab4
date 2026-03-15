import numpy as np


class AddNorm:
    
    def __init__(self, epsilon=1e-6):
      
        self.epsilon = epsilon


    def layer_norm(self, x):

        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        return normalized


    def forward(self, x, sublayer_output):
     
        return self.layer_norm(x + sublayer_output)