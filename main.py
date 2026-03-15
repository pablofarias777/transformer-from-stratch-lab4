import numpy as np

from attention import scaled_dot_product_attention
from ffn import FeedForward

Q = np.random.randn(4,64)
K = np.random.randn(4,64)
V = np.random.randn(4,64)

attn_out = scaled_dot_product_attention(Q,K,V)

ffn = FeedForward()

ffn_out = ffn.forward(attn_out)

print("Attention shape:", attn_out.shape)
print("FFN shape:", ffn_out.shape)