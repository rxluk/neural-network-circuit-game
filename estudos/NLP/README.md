# NLP: Natural Language Processing

## Por Que NLP?

Compreender texto é diferente de imagens ou números. Palavra são **símbolos** com significado semântico.

"gato" ≠ "cão" (diferentes símbolos)
"gato" ≈ "gata" (relacionados semanticamente)

NLP transforma isso em números que redes neurais entendem.

## Trilha de Aprendizado

### [1. Tokenização e Embeddings](./01_tokenizacao_e_embeddings.md) 🔴 CRÍTICO
- Tokenização: como dividir texto em pedaços
- Vocabulário e indexação
- Word2Vec, GloVe, FastText: transformar palavras em vetores
- Embedding layers em redes

**Prerequisito:** Nenhum (conceitos introdutórios)
**Tempo:** 2-3 horas
**Laboratório:** Carregar embeddings pré-treinados, visualizar espaço semântico

### [2. Recurrent Neural Networks (RNNs)](./02_rnns_fundamentals.md) 🔴 CRÍTICO
- Por que sequências requerem RNNs
- Arquitetura vanilla RNN: `hidden[t] = f(input[t], hidden[t-1])`
- LSTM e GRU: soluções para vanishing gradient
- Backpropagation Through Time (BPTT)

**Prerequisito:** [Backpropagation](../AI/NN/03_backprop_derivadas_chain_rule.md), Gradient checking
**Tempo:** 3-4 horas
**Laboratório:** Implementar LSTM simples, testar em sequências

### [3. Attention e Transformers](./03_attention_transformers.md) 🔴 CRÍTICO
- Por que attention? Problema de contexto longo em RNNs
- Self-attention: como tokens "atendem" uns aos outros
- Multi-head attention
- Transformer: stack de atenção + feedforward
- Positional encoding: comunicar posição

**Prerequisito:** RNNs, Álgebra Linear
**Tempo:** 4-5 horas
**Laboratório:** Implementar single-head attention, depois multi-head

### [4. Language Models e Pretraining](./04_language_models_pretraining.md) 🟠 ALTO
- Definição: probabilidade do próximo token `P(w_t | w_{<t})`
- NextToken prediction vs Masked Language Model
- GPT-style vs BERT-style pretraining
- Why pretrain? Transfer learning em NLP

**Prerequisito:** Transformers, Cross-entropy loss
**Tempo:** 2-3 horas
**Laboratório:** Implementar language model simples, testar previsão

### [5. Aplicações Práticas: Classificação e Geração](./05_aplicacoes_nlp.md) 🟠 ALTO
- Text classification: fine-tuning BERT
- Sequence labeling: NER, POS tagging
- Generação: beam search, top-k sampling
- Métrica: BLEU, ROUGE, perplexity

**Prerequisito:** Language models, transformers
**Tempo:** 2-3 horas
**Laboratório:** Fine-tune modelo pré-treinado para tarefa específica

### [6. Desafios Modernos: Contexto, Memória, Interpretabilidade](./06_desafios_modernos.md) 🟡 MÉDIO
- In-context learning em LLMs
- Long-context: sparse attention, recurrence
- Interpretabilidade: attention weights, saliência
- Alinhamento e segurança

**Prerequisito:** Tudo acima
**Tempo:** 2-3 horas
**Laboratório:** Visualizar attention weights, interpretar previsões

## Mapa Rápido

**Quer apenas entender embeddings?** → Leia módulo 1
**Quer implementar chatbot simples?** → Estude 1→2→3→5
**Quer entender GPT/BERT?** → Estude tudo em ordem
**Quer fine-tunar já pronto?** → Comece por 5 (aplicações práticas)

## Checklist de Prontidão

Antes de começar NLP, garanta que você entende:

- [ ] Vetores e matrizes (álgebra linear)
- [ ] Backpropagation (forward + backward)
- [ ] Softmax e cross-entropy
- [ ] Batch normalization e regularização
- [ ] ReLU e outras ativações modernas

Se não tem tudo, revise [Matemática](../matematica/README.md) e [Neural Networks](../AI/NN/README.md).

## Código Laboratório

Cada módulo tem seção **"Código Pronto"** com implementação completa:

```python
# Exemplo: executar num2words tokenizer
from tokenizacao import SimpleTokenizer
tok = SimpleTokenizer()
print(tok.tokenize("Hello world"))  # ['Hello', 'world']
```

Todos os exemplos rodáveis localmente (sem PyTorch/TensorFlow).

## Onde Está o Projeto?

O projeto neural-network-circuit-game usa **neuroevolution** (algoritmo genético), não transformers.

Mas entender NLP:
- Ajuda a entender redes recorrentes (usadas em simulação?)
- Abre porta para multi-modal (visão + linguagem)
- Transferência: padrões aprendidos em NLP → aplicáveis em outras áreas

## Referências

- Goodfellow et al. "Deep Learning" Cap. 9-12
- Vaswani et al. (2017) "Attention is All You Need"
- Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2018) "Language Models are Unsupervised Multitask Learners"
