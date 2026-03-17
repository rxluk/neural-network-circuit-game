# NLP 01: Tokenização e Embeddings

## 1. Tokenização: Quebrando Texto em Pedaços

### 1.1 O Problema

Rede neural trabalha com números, não strings.

```
Input: "Hello, world!"
Output: ???
```

Precisamos converter cada **token** (palavra, caractere, subword) em **número inteiro**.

### 1.2 Estratégias de Tokenização

#### Nível de Caractere

```python
def char_tokenize(text):
    """Cada caractere é token."""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    return [char_to_idx[c] for c in text]


text = "hello"
print(char_tokenize(text))
# Possível output: [2, 1, 2, 2, 3]
```

**Vantagens:** Vocab pequeno, pode gerar qualquer palavra
**Desvantagens:** Sequências longas, difícil aprender semântica

---

#### Nível de Palavra

```python
def word_tokenize(text):
    """Cada palavra é token (split por espaço)."""
    words = text.lower().split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return [word_to_idx[w] for w in words]


text = "the cat sat on the mat"
print(word_tokenize(text))
# Output: [4, 0, 3, 2, 4, 1]
```

**Vantagens:** Sequências curtas, semântica melhor
**Desvantagens:** Vocab grande (50k-100k palavras), palavras desconhecidas (OOV)

---

#### Nível de Subword (BPE - Byte Pair Encoding)

GPT, BERT usam esta abordagem.

**Ideia:** Construir vocabulário iterativamente, mesclando tokens frequentes.

```
Iteração 0: "h e l l o", "w o r l d", "!"
Iteração 1: "he llo", "wor ld", "!"
Iteração 2: "hello", "world", "!"
```

```python
def bpe_tokenize(text, num_merges=100):
    """Simplified BPE."""
    # Inicializar com caracteres
    vocab = set()
    for char in text:
        vocab.add(char)
    
    tokens = list(text)
    
    for _ in range(num_merges):
        # Encontrar pari mais frequente
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs:
            break
        
        most_common_pair = max(pairs, key=pairs.get)
        
        # Mesclar
        new_token = most_common_pair[0] + most_common_pair[1]
        tokens = merge_tokens(tokens, most_common_pair, new_token)
        vocab.add(new_token)
    
    return tokens, vocab


def merge_tokens(tokens, pair, new_token):
    """Replace pair com new_token em tokens."""
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens
```

**Vantagens:** Balanço entre tamanho do vocab (32k tokens) e comprimento das sequências
**Usado em:** GPT (tiktoken), BERT, LLaMA

---

### 1.3 Tratamento de Palavras Desconhecidas (OOV)

```python
class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab = {}
        self.inv_vocab = {}
        self.UNK_TOKEN = "<UNK>"
        self.PAD_TOKEN = "<PAD>"
        self.vocab[self.UNK_TOKEN] = 0
        self.vocab[self.PAD_TOKEN] = 1
        self.vocab_size = vocab_size
    
    def build_vocab(self, texts, max_vocab=None):
        """Construir vocabulário de list de textos."""
        if max_vocab is None:
            max_vocab = self.vocab_size - 2  # Leave room for special tokens
        
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep top K words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab]
        
        for idx, (word, freq) in enumerate(top_words, start=2):
            self.vocab[word] = idx
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        """Texto → lista de índices."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab[self.UNK_TOKEN]))
        return tokens
    
    def decode(self, tokens):
        """Lista de índices → texto."""
        return " ".join([self.inv_vocab.get(idx, self.UNK_TOKEN) for idx in tokens])


# Usar
tokenizer = Tokenizer(vocab_size=100)
tokenizer.build_vocab(["hello world", "hello there", "world peace"])
print(tokenizer.encode("hello world"))  # [2, 3]
print(tokenizer.decode([2, 3]))           # "hello world"
```

---

## 2. Word Embeddings: Números com Significado

### 2.1 O Problema com Codificação One-Hot

```python
# One-hot encoding (naive)
vocab = ["cat", "dog", "bird"]
cat_one_hot = [1, 0, 0]
dog_one_hot = [0, 1, 0]

# Problema: documentos ortogonais
similarity("cat", "dog") = 0  # Completamente desconectados!
similarity("cat", "cat") = 1   # Perfeito
```

Isso não captura que "cat" e "dog" são ambos **animais**.

### 2.2 Embeddings: Espaço Denso

**Ideia:** Representar cada palavra como vetor denso de baixa dimensão.

```python
# Embedding space (dimensão 2, para visualizar)
embedding = {
    "cat": [0.9, 0.1],      # próximo a "dog"
    "dog": [0.85, 0.15],    # próximo a "cat"
    "bird": [0.7, 0.5],     # pássaro (mais longe de mamíferos)
    "king": [0.2, 0.9],     # conceito diferente
    "queen": [0.1, 0.85]    # perto de "king"
}

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Agora similaridades fazem sentido!
print(cosine_similarity(embedding["cat"], embedding["dog"]))     # ~0.99
print(cosine_similarity(embedding["cat"], embedding["bird"]))    # ~0.95
print(cosine_similarity(embedding["cat"], embedding["king"]))    # ~0.18
```

### 2.3 Word2Vec: Aprendendo Embeddings

**Ideia:** Palavras que aparecem em contexto similar devem ter embeddings similares.

**Problema:** Dado texto "The cat sat on the mat", prever palavra de contexto.

```
Input token: "cat"     → Target: "sat" ou "on" (Context words)
Input token: "sat"     → Target: "cat" ou "on"
```

#### Skip-Gram Model

```
Input: one_hot_encoded_word → Embedding layer → Hidden → Output softmax
```

```python
class Word2Vec:
    def __init__(self, vocab_size, embedding_dim):
        # W: palavra → embedding
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        # U: embedding → contexto (predict context)
        self.U = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def forward(self, word_idx, context_idx):
        """
        word_idx: índice da palavra (one-hot em format integer)
        context_idx: índice da palavra de contexto
        """
        # Embedding lookup
        embedding = self.W[word_idx]  # (embedding_dim,)
        
        # Output logits
        logits = embedding @ self.U  # (vocab_size,)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[context_idx] + 1e-9)
        
        return loss, probs, embedding
    
    def backward(self, word_idx, context_idx, learning_rate=0.01):
        loss, probs, embedding = self.forward(word_idx, context_idx)
        
        # Gradient de probabilidades
        grad_logits = probs.copy()
        grad_logits[context_idx] -= 1  # Cross-entropy gradient
        
        # Backprop através OUT → U
        grad_U = np.outer(embedding, grad_logits)
        
        # Backprop através embedding → W
        grad_embedding = self.U @ grad_logits
        grad_W = np.zeros_like(self.W)
        grad_W[word_idx] = grad_embedding
        
        # Update
        self.U -= learning_rate * grad_U
        self.W -= learning_rate * grad_W
        
        return loss


# Exemplo: treinar em corpus simples
vocab = ["the", "cat", "sat", "on", "mat"]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}

w2v = Word2Vec(vocab_size, embedding_dim=4)

# Sentence: "the cat sat on the mat"
sentence_indices = [word2idx[w] for w in ["the", "cat", "sat", "on", "the", "mat"]]

# Training pairs (word, context)
pairs = []
window_size = 2
for i in range(window_size, len(sentence_indices) - window_size):
    target = sentence_indices[i]
    contexts = (
        sentence_indices[i-window_size:i] + 
        sentence_indices[i+1:i+window_size+1]
    )
    for context in contexts:
        pairs.append((target, context))

# Train
for epoch in range(100):
    total_loss = 0
    for word_idx, context_idx in pairs:
        loss = w2v.backward(word_idx, context_idx, learning_rate=0.1)
        total_loss += loss
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(pairs):.4f}")

# Extrair embeddings
embeddings = w2v.W
# Agora embeddings[word2idx["cat"]] contém embedding da palavra "cat"
```

### 2.4 Propriedades Emergentes

Embeddings bem-treinados capturam propriedades semânticas:

```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
embedding("paris") - embedding("france") + embedding("italy") ≈ embedding("rome")
```

Isto é capturado implicitamente durante treinamento!

---

## 3. Modelos Pré-Treinados: GloVe, FastText

### 3.1 GloVe (Global Vectors)

Combina contagem de coocorrência com fatoração de matriz.

```python
# Conceitual: minimizar
# minimize Σ_ij f(X_ij) * (embedding_i · embedding_j - log(X_ij))²
# onde X_ij = frequência de coocorrência palavra i ~ j
```

Propriedade: Mais interpretável (dimensões representam conceitos semânticos)

### 3.2 FastText

Como Word2Vec, mas **subword-aware**.

```
Palavra: "running"
Subwords: "run", "running", "ing"
Embedding("running") = média(embeddings de subwords)
```

Vantagem: Reduz OOV problem (palavras desconhecidas ≈ composição de subwords)

```python
# Pseudocódigo
def fasttext_embedding(word, subword_embeddings, n_grams=3):
    """Compute embedding as mean of subwords."""
    subwords = get_ngrams(word, n_grams)
    return np.mean([subword_embeddings[sw] for sw in subwords], axis=0)
```

---

## 4. Embedding Layer em Redes Neurais

### 4.1 Lookup Table

```python
class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        # Matriz de embeddings
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.embedding_dim = embedding_dim
    
    def forward(self, token_indices):
        """
        token_indices: array de shape (batch,) ou (batch, seq_len)
        Retorna: embeddings shape (..., embedding_dim)
        """
        if len(token_indices.shape) == 1:
            # Single token per sample
            return self.W[token_indices]
        else:
            # Sequence of tokens
            return self.W[token_indices]
    
    def backward(self, grad_output, token_indices):
        """Gradient w.r.t. W."""
        grad_W = np.zeros_like(self.W)
        
        if len(token_indices.shape) == 1:
            np.add.at(grad_W, token_indices, grad_output)
        else:
            for i, idx in enumerate(token_indices.flatten()):
                grad_W[idx] += grad_output.flatten()[i]
        
        return grad_W


# Usar em rede
emb_layer = EmbeddingLayer(vocab_size=1000, embedding_dim=64)
token_ids = np.array([5, 23, 101, 7])   # IDs of 4 tokens
embeddings = emb_layer.forward(token_ids)  # shape (4, 64)
```

### 4.2 Inicializar com Embeddings Pré-Treinados

```python
def load_pretrained_embeddings(embedding_path, tokenizer):
    """Load Word2Vec, GloVe embeddings."""
    embeddings = {}
    with open(embedding_path) as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    
    # Construir matriz
    vocab_size = len(tokenizer.vocab)
    embedding_dim = len(list(embeddings.values())[0])
    W = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in tokenizer.vocab.items():
        if word in embeddings:
            W[idx] = embeddings[word]
        else:
            W[idx] = np.random.randn(embedding_dim) * 0.01
    
    return W
```

---

## 5. Visualizando Embeddings: t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assumindo 'embeddings' de shape (vocab_size, embedding_dim)
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embeddings)

plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
for i, word in enumerate(vocab):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
plt.title("Word Embeddings (t-SNE)")
plt.show()
```

Visualização mostra:
- Palavras similares próximas uma da outra
- Clusters temáticos emergem
- Propriedades semânticas geométricas

---

## 6. Exercícios

### Ex. 1: Build Tokenizer

Construir tokenizer com BPE simples. Tokenizar texto.

### Ex. 2: Train Word2Vec

Treinar embeddings em corpus pequeno. Verificar similaridades.

### Ex. 3: Análise Semântica

Carregar GloVe/Word2Vec pré-treinados. Calcular:
- `king - man + woman`
- `paris - france + italy`

---

## Próximo

[2. RNNs e LSTMs](./02_rnns_fundamentals.md)
