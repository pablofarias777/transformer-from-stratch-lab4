import numpy as np


def create_causal_mask(seq_len):
    
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)

    mask = mask * -1e9

    return mask