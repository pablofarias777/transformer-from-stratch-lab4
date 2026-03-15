import numpy as np

from encoder import EncoderBlock
from decoder import DecoderBlock


class Transformer:

    def __init__(self, d_model=512, d_ff=2048, vocab_size=100):

        self.encoder = EncoderBlock(d_model, d_ff)
        self.decoder = DecoderBlock(d_model, d_ff)

        # projeção final para o vocabulário
        self.Wo = np.random.randn(d_model, vocab_size) * np.sqrt(2 / d_model)
        self.bo = np.zeros((1, vocab_size))


    def softmax(self, x):

        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


    def forward(self, encoder_input, decoder_input):

        # 1️⃣ rodar encoder
        encoder_output = self.encoder.forward(encoder_input)

        # 2️⃣ rodar decoder
        decoder_output = self.decoder.forward(decoder_input, encoder_output)

        # 3️⃣ projeção para vocabulário
        logits = decoder_output @ self.Wo + self.bo

        # 4️⃣ softmax
        probs = self.softmax(logits)

        return probs