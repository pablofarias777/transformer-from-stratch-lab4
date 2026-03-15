import numpy as np
from attention import scaled_dot_product_attention
from add_norm import AddNorm
from ffn import FeedForward


class EncoderBlock:

    def __init__(self, d_model=512, d_ff=2048):


        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()

        self.ffn = FeedForward(d_model, d_ff)


    def forward(self, x):
       
        attn_output, _ = scaled_dot_product_attention(x, x, x)

        x = self.add_norm1.forward(x, attn_output)

    
        ffn_output = self.ffn.forward(x)

        x = self.add_norm2.forward(x, ffn_output)

        return x