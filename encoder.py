import numpy as np
from attention import scaled_dot_product_attention
from add_norm import AddNorm
from ffn import FeedForward


class EncoderBlock:

    def __init__(self, d_model=512, d_ff=2048):
        """
        Um bloco do Encoder do Transformer
        """

        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()

        self.ffn = FeedForward(d_model, d_ff)


    def forward(self, x):
        """
        x : (seq_len, d_model)
        """

        # 1️⃣ Self Attention
        attn_output, _ = scaled_dot_product_attention(x, x, x)

        # 2️⃣ Add & Norm
        x = self.add_norm1.forward(x, attn_output)

        # 3️⃣ Feed Forward
        ffn_output = self.ffn.forward(x)

        # 4️⃣ Add & Norm
        x = self.add_norm2.forward(x, ffn_output)

        return x