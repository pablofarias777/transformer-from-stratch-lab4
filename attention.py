import numpy as np


def softmax(x):

    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    
    d_k = Q.shape[-1]

    scores = Q @ K.T

    scores = scores / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)

    output = attention_weights @ V

    return output, attention_weights