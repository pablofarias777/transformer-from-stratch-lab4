import numpy as np
from attention import scaled_dot_product_attention

Q = np.random.randn(4, 64)
K = np.random.randn(4, 64)
V = np.random.randn(4, 64)

output = scaled_dot_product_attention(Q, K, V)

print("Output shape:", output.shape)