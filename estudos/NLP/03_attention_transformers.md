# NLP 03: Atenção e Transformers

## 1. Problema com RNNs: Contexto Longo

### 1.1 Bottleneck

RNN processa sequência sequencialmente.

```
Input:  "O gato, que estava dormindo em cima da maca, acordou quando..."
RNN:    [token1] → [hidden1]
        [token2] → [hidden2] ← hidden1 perdeu info de token1!
        [token3] → [hidden3] ← hidden2 perdeu info de token1 e token2!
        ...
```

**Problema:** Hidden state é "comprimido". Informação antiga desaparece.

```
Dependência: "gato" (token 1) → "acordou" (token 15)
Distance = 14 tokens

RNN precisa passar hidden state através de 14 camadas!
Gradiente desvanece.
```

### 1.2 Solução: Atenção Direta

**Ideia:** Em vez de passar por hidden state, acessar **diretamente** cada token anterior.

```
Para prever token T:
    - Verificar similaridade com token 1, 2, ..., T-1
    - Ponderar informação baseado em similaridade
    - Combinar tudo
```

Matematicamente: Tokens se "atendem" diretamente.

---

## 2. Self-Attention: O Mecanismo Central

### 2.1 Ideia: Trio de Vetores

Para cada token t:
- **Query** $Q_t$: "O que eu quero saber?"
- **Key** $K_i$: "O que outros tokens oferecem?"
- **Value** $V_i$: "Qual é a informação?"

```
Q: [1, 0, 0, 1, 0]  ← Minha pergunta
K: [1, 1, 0, 0, 0]  ← O que oferece token 1
    [0, 1, 0, 1, 0]  ← O que oferece token 2
    [1, 0, 1, 0, 0]  ← O que oferece token 3
    ...

Similaridade com token 1: Q · K_1 = 1
Similaridade com token 2: Q · K_2 = 0 + 0 + 0 + 1 + 0 = 1
Similaridade com token 3: Q · K_3 = 1
```

### 2.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Onde:
- $Q$: matriz de queries (batch, seq_len, d_k)
- $K$: matriz de keys (batch, seq_len, d_k)
- $V$: matriz de values (batch, seq_len, d_v)
- $d_k$: dimensão das keys (escala para evitar valores very large)

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    Retorna: attention_output (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Similaridade: Q · K^T
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    # Shape: (batch, seq_len, seq_len)
    
    # Softmax w.r.t. keys (última dimensão)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=2, keepdims=True)
    # Shape: (batch, seq_len, seq_len)
    
    # Ponderar values
    output = attention_weights @ V
    # Shape: (batch, seq_len, d_v)
    
    return output, attention_weights


# Exemplo
batch_size, seq_len, d_k, d_v = 2, 4, 8, 8

Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")      # (2, 4, 8)
print(f"Weights shape: {weights.shape}")    # (2, 4, 4)
```

### 2.3 Interpretação Geométrica

```
Attention weight[i, j] = softmax(Q_i · K_j)

Alto: Token i "quer atenção de" token j
Baixo: Token i ignora token j

Exemplo: Pronome "ele" atende a "gato"
    Q_pronome · K_gato = ALTO
```

---

## 3. Masking: Evitar Vazamento de Informação

### 3.1 Causal Masking (Language Model)

Ao prever token T, **não pode ver** tokens T+1, T+2, ...

```
Token 1: pode ver [1]
Token 2: pode ver [1, 2]
Token 3: pode ver [1, 2, 3]  ← NÃO [1, 2, 3, 4, 5]
```

```python
def causal_mask(seq_len):
    """Matriz triangular inferior."""
    return np.tril(np.ones((seq_len, seq_len)))


def apply_mask(scores, mask):
    """scores: (batch, seq_len, seq_len)."""
    scores[mask == 0] = -np.inf
    return scores


# Uso em attention
mask = causal_mask(seq_len=4)
scores_masked = apply_mask(scores, mask)
attention_weights = softmax(scores_masked)
```

---

## 4. Multi-Head Attention: Múltiplas Perspectivas

### 4.1 Problema com Single Head

Com uma única atenção, modelo é **restrito** a uma forma de correlação.

Analogia: Uma pessoa com 1 olho vs 2 olhos
- 1 olho: profundidade limitada
- 2 olhos: múltiplos ângulos

### 4.2 Solução: h Heads Paralelos

```
Para cada head k:
    Q_k = W_Q_k · X
    K_k = W_K_k · X
    V_k = W_V_k · X
    head_k = attention(Q_k, K_k, V_k)

Concatenar: output = concat(head_1, ..., head_h) · W_O
```

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Linear projections para cada head
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq_len, d_model)
        Retorna: (batch, seq_len, d_model)
        """
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = Q @ self.W_Q  # (batch, seq_len, d_model)
        K = K @ self.W_K
        V = V @ self.W_V
        
        # Reshape into multiple heads
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # → (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention em cada head
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # (batch, num_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = apply_mask(scores, mask)
        
        attention_weights = softmax_last_dim(scores)
        head_output = attention_weights @ V
        # (batch, num_heads, seq_len, d_k)
        
        # Concatenar heads
        head_output = head_output.transpose(0, 2, 1, 3)
        # (batch, seq_len, num_heads, d_k)
        head_output = head_output.reshape(batch_size, -1, self.d_model)
        # (batch, seq_len, d_model)
        
        # Final linear projection
        output = head_output @ self.W_O
        
        return output, attention_weights


def softmax_last_dim(x):
    """Softmax na última dimensão."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 4.3 Por Que Funciona

Pensemos em uma sequência de sentenças:

```
Head 1: Atenção a adjacência (context local)
Head 2: Atenção a palavras-chave distantes (anáfora)
Head 3: Atenção a características gramaticais
...
```

Automaticamente aprendido durante treino!

---

## 5. Transformer: Stack de Atenção + Feedforward

### 5.1 Arquitetura Completa

```
Input Embedding + Positional Encoding
↓
Encoder (N camadas):
    ├─ Multi-Head Attention
    ├─ Layer Norm + Residual
    ├─ Feedforward
    └─ Layer Norm + Residual
↓
(só para decoder)
Decoder (N camadas):
    ├─ Causal Self-Attention
    ├─ Encoder-Decoder Attention
    ├─ Feedforward
    └─ Layer Norms + Residuals
↓
Output Linear → Softmax
```

### 5.2 Positional Encoding

RNNs / Transformers com atenção **não sabem ordem**!

```
"dog bites man" ≠ "man bites dog"

Mas ambos com mesma atenção dão MESMA resposta!
```

Solução: Adicionar **posição** como feature.

```
Transformers usam sin/cos ondas:
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

```python
def positional_encoding(seq_len, d_model):
    """Gerar positional encoding."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# Exemplo
pe = positional_encoding(seq_len=100, d_model=64)
# Visualizar
import matplotlib.pyplot as plt
plt.imshow(pe[:50, :50], cmap='viridis')
plt.xlabel('d_model')
plt.ylabel('Position')
plt.title('Positional Encoding')
plt.colorbar()
plt.show()
```

Propriedade: PE(pos + k) pode ser expressa como linear combination de PE(pos)
→ Modelo aprende "saltar" de posição automaticamente.

### 5.3 Transformer Encoder

```python
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feedforward network: d_model → d_ff → d_model
        self.W_1 = np.random.randn(d_model, d_ff) * 0.01
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.01
        self.b_2 = np.zeros(d_model)
        
        # Layer normalization (scale + shift after zero-mean)
        self.gamma_attn = np.ones(d_model)
        self.beta_attn = np.zeros(d_model)
        self.gamma_ff = np.ones(d_model)
        self.beta_ff = np.zeros(d_model)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # Self-attention
        attn_output, _ = self.attention.forward(x, x, x)
        
        # Residual + Layer Norm
        x = x + attn_output
        x = self.layer_norm(x, self.gamma_attn, self.beta_attn)
        
        # Feedforward
        ff_output = x @ self.W_1 + self.b_1
        ff_output = np.maximum(0, ff_output)  # ReLU
        ff_output = ff_output @ self.W_2 + self.b_2
        
        # Residual + Layer Norm
        x = x + ff_output
        x = self.layer_norm(x, self.gamma_ff, self.beta_ff)
        
        return x
    
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pe = positional_encoding(seq_len=1000, d_model=d_model)
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
    
    def forward(self, token_indices):
        """
        token_indices: (batch, seq_len)
        """
        seq_len = token_indices.shape[1]
        
        # Embedding + Positional encoding
        x = self.embedding[token_indices]  # (batch, seq_len, d_model)
        x = x + self.pe[:seq_len]
        
        # Apply layers
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
```

---

## 6. Comparação: RNN vs Transformer

| Aspecto | RNN/LSTM | Transformer |
|---------|----------|-------------|
| Parâmetros | 3x (gates) | ~igual |
| Dependência longa | Má (vanishing) | Ótima (atenção direta) |
| Paralelização | Sequencial ✗ | Paralelo ✓ |
| Velocidade | Lento | Rápido (GPUs) |
| Memória (seq_len) | O(1) | **O(seq_len²)** ← problema para seqs muito longas |
| SOTA | Obsoleto | ✓✓✓ |

---

## 7. Exercícios

### Ex. 1: Implementar Single-Head Attention

Completo forward + backward.

### Ex. 2: Visualizar Attention Weights

Train transformer simples, plotar weights para visualizar quais tokens se atendem.

### Ex. 3: Comparação RNN vs Transformer

Train ambos em sequências longas (100+ tokens). Qual aprende melhor?

---

## Próximo

[4. Language Models e Pretraining](./04_language_models_pretraining.md)
