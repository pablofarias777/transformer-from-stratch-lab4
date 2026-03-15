import numpy as np

from attention import scaled_dot_product_attention
from ffn import FeedForward
from add_norm import AddNorm
from encoder import EncoderBlock


def test_encoder():

    print("\n--- Testing Encoder ---")

    x = np.random.rand(4, 512)

    encoder = EncoderBlock()

    output = encoder.forward(x)

    print("Encoder output shape:", output.shape)


def test_attention():

    print("\n--- Testing Attention ---")

    Q = np.random.rand(4, 512)
    K = np.random.rand(4, 512)
    V = np.random.rand(4, 512)

    output, attention = scaled_dot_product_attention(Q, K, V)

    print("Attention output shape:", output.shape)
    print("Attention weights shape:", attention.shape)


def test_ffn():

    print("\n--- Testing Feed Forward ---")

    x = np.random.rand(4, 512)

    ffn = FeedForward()

    output = ffn.forward(x)

    print("FFN output shape:", output.shape)


def test_add_norm():

    print("\n--- Testing Add & Norm ---")

    x = np.random.rand(4, 512)
    sublayer_output = np.random.rand(4, 512)

    add_norm = AddNorm()

    output = add_norm.forward(x, sublayer_output)

    print("AddNorm output shape:", output.shape)


if __name__ == "__main__":

    print("Running Transformer component tests")

    test_attention()
    test_ffn()
    test_add_norm()
    test_encoder()