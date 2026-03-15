# Transformer Encoder-Decoder From Scratch

Projeto desenvolvido para o **Laboratório 4 da disciplina Tópicos em Inteligência Artificial**.

O objetivo deste trabalho é implementar a arquitetura completa do **Transformer Encoder-Decoder** utilizando apenas **Python e NumPy**, sem o uso de bibliotecas de deep learning como PyTorch ou TensorFlow.

O projeto integra todos os componentes implementados nos laboratórios anteriores e executa uma simulação de geração de sequência utilizando **inferência auto-regressiva**.

---

# Estrutura do projeto

Arquivos principais:

main.py → executa o modelo e simula a inferência auto-regressiva  
attention.py → implementação do Scaled Dot Product Attention  
add_norm.py → implementação da conexão residual com Layer Normalization  
ffn.py → implementação da Feed Forward Network (FFN)  
encoder.py → implementação do EncoderBlock  
decoder.py → implementação do DecoderBlock  
transformer.py → implementação do modelo Transformer completo (Encoder + Decoder)  
utils.py → funções auxiliares como criação da máscara causal  

---

# O que foi implementado

Este projeto implementa os principais componentes da arquitetura Transformer.

## 1. Scaled Dot Product Attention

Implementação da atenção baseada no produto escalar escalado entre **Queries, Keys e Values**, responsável por calcular a relevância entre tokens da sequência.

---

## 2. Add & Norm (Residual Connection + Layer Normalization)

Implementação da conexão residual seguida de normalização, que ajuda a estabilizar o treinamento e facilitar o fluxo de gradientes na rede.

---

## 3. Feed Forward Network (FFN)

Rede neural totalmente conectada aplicada posição por posição na sequência.

Estrutura utilizada:

Linear → ReLU → Linear

com expansão de dimensão:

d_model → d_ff → d_model

---

## 4. Encoder Block

Implementação do bloco do Encoder seguindo o fluxo:

Input  
↓  
Self Attention  
↓  
Add & Norm  
↓  
Feed Forward Network  
↓  
Add & Norm  

O Encoder produz uma representação contextualizada da sequência de entrada.

---

## 5. Decoder Block

Implementação do bloco do Decoder contendo:

Masked Self Attention  
↓  
Add & Norm  
↓  
Cross Attention (consulta a saída do Encoder)  
↓  
Add & Norm  
↓  
Feed Forward Network  
↓  
Add & Norm  

O Decoder utiliza a informação do Encoder para gerar a sequência de saída.

---

## 6. Máscara Causal (Masked Self Attention)

Implementação de uma **máscara triangular inferior** que impede o modelo de acessar tokens futuros durante a geração de texto.

Isso garante que a geração seja **auto-regressiva**.

---

## 7. Loop de geração auto-regressiva

O modelo simula a geração de uma sequência iniciando com o token:

<START>

A cada iteração o modelo prevê o próximo token, que é adicionado à sequência até que seja gerado:

<EOS>

ou até atingir um número máximo de passos.

---

# Como executar o projeto

## 1. Instalar dependências

O projeto utiliza apenas NumPy.

pip install numpy

---

## 2. Rodar o programa

No Mac ou Linux

python3 main.py

No Windows

python main.py

---

# Saída esperada

O programa deve produzir algo semelhante a:

Running Transformer Inference

Step 1: predicted -> token_4  
Step 2: predicted -> token_9  
Step 3: predicted -> token_4  
Step 4: predicted -> token_7  
Step 5: predicted -> token_15  

Generated sequence:  
['<START>', 'token_4', 'token_9', 'token_4', 'token_7', 'token_15']

Essa saída demonstra que:

- O **Encoder processou a sequência de entrada**
- O **Decoder gerou tokens utilizando atenção**
- O **loop auto-regressivo funcionou corretamente**
