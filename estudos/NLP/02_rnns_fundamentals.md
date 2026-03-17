# NLP 02: Recurrent Neural Networks (RNNs)

## 1. Por Que Redes Recorrentes?

### 1.1 Problema com Feedforward

Rede feedforward padrão:

```
Input: [5, 23, 101, 7]  (4 tokens)
Dense layers
Output: classe (p.ex. "positivo")
```

Problema: **Sequência é flattened**, ordem não importa.

```
Mesma predição para: [5, 23, 101, 7] e [7, 101, 23, 5]
Ambas são malucas para texto! Ordem IMPORTA.
```

### 1.2 RNN Vanilla: Memória Recorrente

**Ideia:** Manter **hidden state** que é atualizado em cada timestep.

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Onde:
- $x_t$: token no tempo t
- $h_t$: hidden state (memória)
- $W_{xh}, W_{hh}, W_{hy}$: pesos

```python
class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # x → hidden
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_h = np.zeros(hidden_dim)
        
        # hidden → hidden
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
        # hidden → output
        self.W_hy = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b_y = np.zeros(output_dim)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x_sequence, h_init=None):
        """
        x_sequence: shape (seq_len, batch_size, input_dim)
        Retorna: 
            - outputs: (seq_len, batch_size, output_dim)
            - hidden_states: lista de h em cada tempo
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        if h_init is None:
            h = np.zeros((batch_size, self.hidden_dim))
        else:
            h = h_init.copy()
        
        outputs = []
        hidden_states = [h.copy()]
        
        for t in range(seq_len):
            x_t = x_sequence[t]  # (batch_size, input_dim)
            
            # h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
            h = np.tanh(
                x_t @ self.W_xh + 
                h @ self.W_hh + 
                self.b_h
            )
            
            # y_t = W_hy h_t + b_y
            y_t = h @ self.W_hy + self.b_y
            
            outputs.append(y_t)
            hidden_states.append(h.copy())
        
        outputs = np.array(outputs)
        return outputs, hidden_states
    
    def backward_through_time(self, x_sequence, hidden_states, 
                              grad_outputs, learning_rate=0.01):
        """
        BPTT: Backprop Through Time
        grad_outputs: gradient w.r.t. each output (seq_len, batch_size, output_dim)
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        # Zeros para gradientes
        grad_W_xh = np.zeros_like(self.W_xh)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_W_hy = np.zeros_like(self.W_hy)
        grad_b_h = np.zeros_like(self.b_h)
        grad_b_y = np.zeros_like(self.b_y)
        
        # Começar do final e trabalhar para trás
        grad_h_next = np.zeros((batch_size, self.hidden_dim))
        
        for t in reversed(range(seq_len)):
            x_t = x_sequence[t]
            h_t = hidden_states[t + 1]  # h após processo em tempo t
            h_prev = hidden_states[t]
            
            # Gradiente do output
            grad_W_hy += h_t.T @ grad_outputs[t]
            grad_b_y += np.sum(grad_outputs[t], axis=0)
            
            # Gradiente w.r.t. h_t
            grad_h = grad_outputs[t] @ self.W_hy.T + grad_h_next
            
            # Gradiente tanh: d/dz tanh(z) = 1 - tanh²(z)
            grad_h_raw = grad_h * (1 - h_t ** 2)
            
            # Gradiente dos pesos
            grad_W_xh += x_t.T @ grad_h_raw
            grad_W_hh += h_prev.T @ grad_h_raw
            grad_b_h += np.sum(grad_h_raw, axis=0)
            
            # Gradiente para timestep anterior
            grad_h_next = grad_h_raw @ self.W_hh.T
        
        # Update weights
        self.W_xh -= learning_rate * grad_W_xh
        self.W_hh -= learning_rate * grad_W_hh
        self.W_hy -= learning_rate * grad_W_hy
        self.b_h -= learning_rate * grad_b_h
        self.b_y -= learning_rate * grad_b_y


# Exemplo: Sequência de 5 timestamps
seq_len, batch_size, input_dim, output_dim = 5, 3, 10, 2
hidden_dim = 8

rnn = VanillaRNN(input_dim, hidden_dim, output_dim)

# Sequência de embeddings
x_seq = np.random.randn(seq_len, batch_size, input_dim)

# Forward
outputs, hidden_states = rnn.forward(x_seq)
print(f"Outputs shape: {outputs.shape}")  # (5, 3, 2)
```

---

## 2. Problema: Vanishing Gradient em RNNs

### 2.1 Análise

BPTT requer calcular:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$

Onde:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=0}^{k-1} \frac{\partial h_{t-i}}{\partial h_{t-i-1}} = \prod_{i=0}^{k-1} W_{hh}^T (1 - h_{t-i}^2)$$

Isto é um **produto de muitos termos menores que 1**, exponencialmente pequeno!

```python
# Demonstração numérica
np.random.seed(0)

# Simular gradiente backproping através de 20 timesteps
T = 20
grad = 1.0
grads = [grad]

for t in range(T):
    # Multiplicar por max eigenvalue of W_hh (assume tanh, max derivada = 1)
    grad *= 0.9  # Simular decaimento
    grads.append(grad)

import matplotlib.pyplot as plt
plt.plot(grads)
plt.xlabel("Timestep (indo para trás)")
plt.ylabel("Gradiente")
plt.title("Vanishing Gradient em RNN (T=20)")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# Output: gradiente cai de 1.0 para ~0.000000001
# Primeiros timesteps não aprendem!
```

### 2.2 Solução: LSTM (Long Short-Term Memory)

**Ideia:** Adicionar **cell state** $C_t$ que é atualizado de forma **aditiva** (não multiplicativa).

```
Multiplicação (RNN)    → Vanishing
Adição (LSTM)         → Mantém gradientes!
```

---

## 3. LSTM: Memória Longa com Portões

### 3.1 Arquitetura

```
Forget gate: f_t = σ(W_f x_t + U_f h_{t-1} + b_f)    # Esquecer quanto?
Input gate:  i_t = σ(W_i x_t + U_i h_{t-1} + b_i)    # Adicionar quanto?
Candidate:   c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)  # O quê adicionar?
Cell state:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ c̃_t         # ADITIVO!
Output gate: o_t = σ(W_o x_t + U_o h_{t-1} + b_o)    # Mostrar quanto?
Hidden:      h_t = o_t ⊙ tanh(C_t)                     # Output final
```

Equações em NumPy:

```python
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Forget gate weights
        self.W_f = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_f = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros(hidden_dim)
        
        # Input gate weights
        self.W_i = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_i = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros(hidden_dim)
        
        # Candidate weights
        self.W_c = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_c = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_c = np.zeros(hidden_dim)
        
        # Output gate weights
        self.W_o = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_o = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros(hidden_dim)
    
    def forward(self, x_t, h_prev, C_prev):
        """
        x_t: input no tempo t (batch_size, input_dim)
        h_prev: hidden state anterior (batch_size, hidden_dim)
        C_prev: cell state anterior (batch_size, hidden_dim)
        """
        # Forget gate
        f_t = 1 / (1 + np.exp(-(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)))
        
        # Input gate
        i_t = 1 / (1 + np.exp(-(x_t @ self.W_i + h_prev @ self.U_i + self.b_i)))
        
        # Candidate
        c_tilde = np.tanh(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        
        # Cell state (KEY: ADITIVO)
        C_t = f_t * C_prev + i_t * c_tilde
        
        # Output gate
        o_t = 1 / (1 + np.exp(-(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)))
        
        # Hidden state
        h_t = o_t * np.tanh(C_t)
        
        return h_t, C_t, (f_t, i_t, c_tilde, o_t, C_t)
    
    def backward(self, x_t, h_prev, C_prev, grad_h_t, grad_C_t, 
                 cache, learning_rate=0.01):
        """Backward pass (simplified)."""
        f_t, i_t, c_tilde, o_t, C_t = cache
        
        # Gradient w.r.t. o_t
        grad_o_t = grad_h_t * np.tanh(C_t)
        
        # Gradient w.r.t. C_t (antes de o_t)
        grad_C_t_add = grad_h_t * o_t * (1 - np.tanh(C_t)**2)
        grad_C_t = grad_C_t + grad_C_t_add
        
        # Gradient w.r.t. f_t, i_t, c_tilde
        grad_f_t = grad_C_t * C_prev
        grad_i_t = grad_C_t * c_tilde
        grad_c_tilde = grad_C_t * i_t
        
        # Propagate to previous cell
        grad_C_prev = grad_C_t * f_t
        
        # Update weights (simplified - não fazer cada um)
        # self.W_f -= learning_rate * gradient
        # ...
        
        return grad_C_prev, (grad_f_t, grad_i_t, grad_c_tilde, grad_o_t)
```

### 3.2 Por que LSTM funciona?

```
Gradiente de C_t para C_{t-1}:
dC_t / dC_{t-1} = f_t (+ outras contribuições)

f_t é sigmóide = entre [0, 1]

Se f_t ≈ 1 (forget gate ativo):
Gradiente = 1, não desaparece!

Se f_t ≈ 0 (esquecer):
Gradiente = 0, mas INTENCIONALMENTE!
```

---

## 4. GRU: LSTM Simplificado

**Ideia:** LSTM com 3 gates é complexo. GRU usa só 2 gates.

```
Reset gate:   r_t = σ(W_r x_t + U_r h_{t-1} + b_r)        # Quanta memória antiga?
Update gate:  z_t = σ(W_z x_t + U_z h_{t-1} + b_z)        # Quanto atualizar?
Candidate:    h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
Hidden:       h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t      # Combinação
```

```python
class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Reset gate
        self.W_r = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_r = np.zeros(hidden_dim)
        
        # Update gate
        self.W_z = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_z = np.zeros(hidden_dim)
        
        # Candidate
        self.W_h = np.random.randn(input_dim, hidden_dim) * 0.01
        self.U_h = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros(hidden_dim)
    
    def forward(self, x_t, h_prev):
        # Reset gate
        r_t = 1 / (1 + np.exp(-(x_t @ self.W_r + h_prev @ self.U_r + self.b_r)))
        
        # Update gate
        z_t = 1 / (1 + np.exp(-(x_t @ self.W_z + h_prev @ self.U_z + self.b_z)))
        
        # Candidate
        h_tilde = np.tanh(x_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
        
        # Hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t, (r_t, z_t, h_tilde)
```

**GRU vs LSTM:**
- GRU: Mais rápido, ~50% menos parâmetros
- LSTM: Maior capacidade, melhor em longas dependências

Na prática: Comece com LSTM, use GRU se estiver lento.

---

## 5. Aplicação: Classificação de Sentimentos

```python
class SentimentClassifier:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM
        self.lstm = LSTMCell(embedding_dim, hidden_dim)
        
        # Output layer
        self.W_out = np.random.randn(hidden_dim, num_classes) * 0.01
        self.b_out = np.zeros(num_classes)
    
    def forward(self, token_sequence):
        """
        token_sequence: (seq_len, batch_size) de indices
        """
        seq_len, batch_size = token_sequence.shape
        embedding_dim = self.embedding.shape[1]
        hidden_dim = self.lstm.hidden_dim
        
        # Embedding lookup
        x_emb = self.embedding[token_sequence]  # (seq_len, batch_size, embedding_dim)
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, hidden_dim))
        C = np.zeros((batch_size, hidden_dim))
        
        caches = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x_emb[t]
            h, C, cache = self.lstm.forward(x_t, h, C)
            caches.append(cache)
        
        # Por último hidden state para classificação
        logits = h @ self.W_out + self.b_out  # (batch_size, num_classes)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs, h, C

# Usar
tokenizer_vocab_size = 10000
classifier = SentimentClassifier(
    vocab_size=tokenizer_vocab_size,
    embedding_dim=64,
    hidden_dim=128,
    num_classes=2  # positive / negative
)

# Sentença tokenizada
tokens = np.array([[5, 10, 23, 7], [3, 11, 1, 2]]).T  # (4, 2) = 4 tokens, 2 sentenças
probs, h, C = classifier.forward(tokens)
print(f"Probs shape: {probs.shape}")  # (2, 2)
```

---

## 6. Checklist: RNN vs LSTM vs GRU

| Aspecto | RNN Vanilla | LSTM | GRU |
|---------|-------------|------|-----|
| Parâmetros | Poucos | 4x mais | 3x mais |
| Vanishing gradient | ✗ Problema | ✓ Resolvido | ✓ Resolvido |
| Dependências longas | Ruins (< 10 steps) | Boas (100+ steps) | Boas |
| Velocidade | Rápido | Lento | Mais rápido |
| Quando usar? | Prototipagem | Produção | Trade-off |

---

## 7. Exercícios

### Ex. 1: Implementar BPTT

Completar backward pass do VanillaRNN.

### Ex. 2: Comparar Architectures

Train RNN vs LSTM vs GRU no mesmo dataset. Plotar converência.

### Ex. 3: Sentiment Classification

Usar classifier para treinar em dataset Real (p.ex. MovieLens).

---

## Próximo

[3. Atenção e Transformers](./03_attention_transformers.md)
