# Redes Neurais 06: Verificação de Gradientes e Debugging

## 1. O Problema: Gradientes Errados = Desastre Silencioso

Seu código roda, loss diminui... mas está aprendendo padrão errado ou não converge bem.

**Causa mais comum:** Bug na implementação do backward pass.

**Por que é difícil encontrar?** Loss pode diminuir com qualquer gradiente (até errado), especialmente em épocas iniciais.

---

## 2. Numerical Gradient Checking: Seu Detector de Bugs

### 2.1 Ideia Central

Comparar gradientes **analíticos** (via backprop) com gradientes **numéricos** (via diferenças finitas).

Se forem iguais (dentro de tolerância), backprop está correto.

### 2.2 Gradiente Numérico (Central Difference)

$$\frac{\partial f}{\partial w} \approx \frac{f(w + \epsilon) - f(w - \epsilon)}{2\epsilon}$$

onde $\epsilon \approx 1e-5$ (pequeno, mas não tão pequeno que sofra underflow).

```python
def numerical_gradient(f, w, eps=1e-5):
    """
    f: função que retorna escalar (loss)
    w: peso ou bias (array)
    Retorna: gradiente numérico com mesma shape de w
    """
    grad = np.zeros_like(w)
    
    # Iterar sobre cada elemento
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        # Salvar valor original
        original = w[idx]
        
        # f(w + epsilon)
        w[idx] = original + eps
        f_plus = f()
        
        # f(w - epsilon)
        w[idx] = original - eps
        f_minus = f()
        
        # Gradiente central
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        # Restaurar
        w[idx] = original
        it.iternext()
    
    return grad
```

### 2.3 Verificação: Comparar Gradientes

```python
def gradient_check(network, x, y_true, threshold=1e-7):
    """
    Verifica se gradientes analíticos (backprop) ≈ numéricos.
    
    Retorna:
    - max_relative_error: máximo erro relativo
    - all_close: bool, se erro < threshold
    """
    batch_size = len(y_true)
    
    # Forward pass
    predictions = network.forward(x, training=False)
    
    # Backward pass (analítico)
    network.backward(y_true)
    analytical_grads = []
    for w in network.weights:
        analytical_grads.append(w.grad.copy())
    
    # Gradientes numéricos
    numerical_grads = []
    
    for layer_idx, w in enumerate(network.weights):
        def loss_fn():
            pred = network.forward(x, training=False)
            eps = 1e-7
            return -np.mean(np.sum(y_true * np.log(pred + eps), axis=1))
        
        num_grad = numerical_gradient(loss_fn, w, eps=1e-5)
        numerical_grads.append(num_grad)
    
    # Comparar
    max_relative_error = 0
    for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
        # Erro relativo: |a - n| / (|a| + |n| + eps)
        rel_error = np.abs(analytical - numerical) / (
            np.abs(analytical) + np.abs(numerical) + 1e-8
        )
        max_rel = np.max(rel_error)
        print(f"Layer {i}: max relative error = {max_rel:.4e}")
        max_relative_error = max(max_relative_error, max_rel)
    
    is_correct = max_relative_error < threshold
    print(f"\n✓ Gradiente verificado: {is_correct}")
    
    return {
        'max_error': max_relative_error,
        'passed': is_correct,
        'threshold': threshold
    }


# Exemplo
class TinyNetwork:
    def __init__(self):
        self.W1 = np.random.randn(10, 5) * 0.1
        self.b1 = np.zeros(5)
        self.W2 = np.random.randn(5, 3) * 0.1
        self.b2 = np.zeros(3)
    
    def forward(self, x, training=True):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        # Softmax
        exp_logits = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.a2 = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return self.a2
    
    def backward(self, y_true):
        batch_size = len(y_true)
        
        # Output layer gradient
        dz2 = self.a2 - y_true  # d(loss)/dz2
        self.dW2 = self.a1.T @ dz2 / batch_size
        self.db2 = np.sum(dz2, axis=0) / batch_size
        
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        self.dW1 = x.T @ dz1 / batch_size
        self.db1 = np.sum(dz1, axis=0) / batch_size
        
        # Store gradients
        self.W1.grad = self.dW1
        self.b1.grad = self.db1
        self.W2.grad = self.dW2
        self.b2.grad = self.db2


# Testar
x_test = np.random.randn(16, 10)
y_test = np.eye(3)[np.random.randint(0, 3, 16)]

net = TinyNetwork()
gradient_check(net, x_test, y_test)
```

---

## 3. Vanishing e Exploding Gradients: Diagnose Cedo

### 3.1 Vanishing Gradient

Gradientes ficam cada vez menores ao retropropagar.

Sintomas:
- Primeiras camadas não aprendem (Loss reduz devagar)
- Pesos não mudam (antes e depois de epoch são iguais)

```python
def diagnose_gradient_flow(network, x, y_true):
    """Print gradient magnitude em cada layer."""
    network.forward(x, training=False)
    network.backward(y_true)
    
    for i, (w, dw) in enumerate(zip(network.weights, network.weight_grads)):
        grad_magnitude = np.mean(np.abs(dw))
        weight_magnitude = np.mean(np.abs(w))
        ratio = grad_magnitude / (weight_magnitude + 1e-8)
        print(f"Layer {i}: ||grad|| = {grad_magnitude:.4e}, weight = {weight_magnitude:.4e}, ratio = {ratio:.4e}")
```

### 3.2 Causas e Soluções

| Problema | Causa | Solução |
|----------|-------|--------|
| Vanishing | Sigmoid/tanh em rede profunda | Use ReLU, GELU; Batch Norm |
| Exploding | Pesos inicializados grandes | Use Xavier/He init |
| Exploding | Learning rate alto | Gradient clipping |

### 3.3 Gradient Clipping

For RNNs e transformers (especialmente suscetíveis a exploding):

```python
def clip_gradients(weights, max_norm=1.0):
    """Clip gradients to have norm <= max_norm."""
    total_norm = np.sqrt(sum(np.sum(w.grad**2) for w in weights))
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for w in weights:
            w.grad *= scale
    
    return total_norm
```

### 3.4 Inicialização Correta

**Xavier Initialization** (para sigmoid/tanh):

$$w \sim \text{Uniform}\left[ -\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \right]$$

**He Initialization** (para ReLU):

$$w \sim \text{Normal}\left(0, \sqrt{\frac{2}{n_{\text{in}}}} \right)$$

```python
def xavier_init(layer_size_in, layer_size_out):
    limit = np.sqrt(6 / (layer_size_in + layer_size_out))
    return np.random.uniform(-limit, limit, (layer_size_in, layer_size_out))


def he_init(layer_size_in):
    return np.random.normal(0, np.sqrt(2 / layer_size_in), layer_size_in)


# Usar
W1 = he_init(784)  # Para ReLU
```

---

## 4. Debugging Workflow: Passo a Passo

### Checklist:

```
[ ] 1. Executar gradient check (numérico vs analítico)
      Se falhar: BUG EM BACKPROP
      
[ ] 2. Plotar loss ao longo das épocas
      Loss deve diminuir monotonicamente (com ruído)
      Se aumentar: learning rate muito alto
      
[ ] 3. Verificar magnitude dos gradientes
      print(f"Mean |grads| = {np.mean(np.abs(grads)):.4e}")
      Muito perto de 0? Vanishing
      Muito grande (> 1)? Exploding
      
[ ] 4. Verificar escala de ativações
      Mean/std de cada layer deve ser razoável
      print(f"Layer {i}: mean={np.mean(a):.2f}, std={np.std(a):.2f}")
      
[ ] 5. Training vs Validation accuracy
      Gap grande? Overfitting → aumentar regularização
      Ambas tão ruins? Underfitting → aumentar capacidade
```

---

## 5. Exemplo Completo: Debugging Sessão

```python
import numpy as np
import matplotlib.pyplot as plt


class DebugNetwork:
    def __init__(self, layer_sizes, init_type='he'):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            if init_type == 'he':
                w = np.random.normal(0, np.sqrt(2 / layer_sizes[i]), 
                                    (layer_sizes[i], layer_sizes[i+1]))
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        a = x
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                # Softmax
                z_shifted = z - np.max(z, axis=1, keepdims=True)
                a = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, y_true, learning_rate=0.01):
        batch_size = len(y_true)
        
        # Output gradient
        delta = self.activations[-1] - y_true
        
        # Backpropagate
        for i in reversed(range(len(self.weights))):
            grad_w = self.activations[i].T @ delta / batch_size
            grad_b = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Update
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.z_values[i-1] > 0)
    
    def loss(self, predictions, y_true):
        eps = 1e-7
        return -np.mean(np.sum(y_true * np.log(predictions + eps), axis=1))


# Dataset simulado
np.random.seed(42)
X_train = np.random.randn(100, 20)
y_train = np.eye(3)[np.random.randint(0, 3, 100)]

X_test = np.random.randn(20, 20)
y_test = np.eye(3)[np.random.randint(0, 3, 20)]

# Train
net = DebugNetwork([20, 64, 32, 3], init_type='he')

losses = []
for epoch in range(50):
    pred_train = net.forward(X_train)
    loss = net.loss(pred_train, y_train)
    losses.append(loss)
    
    net.backward(y_train, learning_rate=0.01)
    
    if epoch % 10 == 0:
        pred_test = net.forward(X_test)
        test_loss = net.loss(pred_test, y_test)
        print(f"Epoch {epoch}: train loss = {loss:.4f}, test loss = {test_loss:.4f}")

# Plot
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. Red Flags: Quando Algo Está Errado

| Sintoma | Provável Causa |
|---------|----------------|
| Loss = NaN | Exploding gradient, learning rate muito alto, bug numérico |
| Loss diminui, mas accuracy não melhora | Bug em métrica, gradientes errados |
| Loss fica flat | Learning rate muito baixo, stuck em local mínimo |
| Train accuracy 100%, test 50% | Overfitting severo, regularização insuficiente |
| Mesmo loss com/sem seu código novo | Código novo não está sendo usado |

---

## 7. Exercícios

### Ex. 1: Encontrar o Bug

Código com backprop errado propositalmente. Use gradient checking para encontrar.

### Ex. 2: Comparar Inicializações

Train com random, xavier, he. Qual converge melhor?

### Ex. 3: Gradient Clipping

Treinar LSTM (ou RNN simples). Plotar ||grad|| com e sem clipping.

---

## Próximo

[→ NLP: Tokenização e Embeddings](../../../estudos/NLP/01_tokenizacao_e_embeddings.md)
