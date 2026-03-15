import numpy as np
from utils import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):

    dk = Q.shape[-1]

    scores = Q @ K.T / np.sqrt(dk)

    if mask is not None:
        scores += mask

    weights = softmax(scores)

    output = weights @ V

    return output