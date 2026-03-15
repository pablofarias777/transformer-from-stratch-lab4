import numpy as np
from attention import scaled_dot_product_attention
from add_norm import AddNorm
from ffn import FeedForward
from utils import create_causal_mask


class DecoderBlock:

    def __init__(self, d_model=512, d_ff=2048):

        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        self.add_norm3 = AddNorm()

        self.ffn = FeedForward(d_model, d_ff)


    def forward(self, y, encoder_output):
    
        seq_len = y.shape[0]

        
        mask = create_causal_mask(seq_len)

        attn_output, _ = scaled_dot_product_attention(y, y, y, mask)

        y = self.add_norm1.forward(y, attn_output)

        cross_output, _ = scaled_dot_product_attention(
            y, encoder_output, encoder_output
        )
        y = self.add_norm2.forward(y, cross_output)

        ffn_output = self.ffn.forward(y)

        y = self.add_norm3.forward(y, ffn_output)

        return y