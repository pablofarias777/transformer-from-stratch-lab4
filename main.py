import numpy as np
from transformer import Transformer


def autoregressive_inference():

    print("Running Transformer Inference")

    d_model = 512
    vocab_size = 20
    max_steps = 10

    # vocabulário simples para simulação
    vocab = {
        "<START>": 0,
        "<EOS>": 1,
        "Thinking": 2,
        "Machines": 3
    }

    inv_vocab = {v: k for k, v in vocab.items()}

    # simular entrada do encoder ("Thinking Machines")
    seq_len = 2
    encoder_input = np.random.rand(seq_len, d_model)

    # criar modelo
    model = Transformer(d_model=d_model, vocab_size=vocab_size)

    # decoder começa com <START>
    decoder_input = np.random.rand(1, d_model)

    generated_tokens = ["<START>"]

    for step in range(max_steps):

        probs = model.forward(encoder_input, decoder_input)

        # pegar última previsão
        last_token_probs = probs[-1]

        predicted_token = np.argmax(last_token_probs)

        # converter para palavra (se existir)
        token_word = inv_vocab.get(predicted_token, f"token_{predicted_token}")

        generated_tokens.append(token_word)

        print(f"Step {step+1}: predicted -> {token_word}")

        # parar se gerar <EOS>
        if token_word == "<EOS>":
            break

        # criar embedding aleatório para novo token
        new_embedding = np.random.rand(1, d_model)

        # concatenar ao decoder input
        decoder_input = np.vstack([decoder_input, new_embedding])

    print("\nGenerated sequence:")
    print(generated_tokens)


if __name__ == "__main__":
    autoregressive_inference()