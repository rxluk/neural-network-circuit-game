# Redes Neurais 05: Ativações Modernas e Regularização

## 1. Ativações: Não Apenas Sigmoid

### 1.1 Por Que Múltiplas Ativações?

**Sigmoid** (usada classicamente):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Problemas:
- **Vanishing gradient:** Deriva $\sigma'(z) siempre \leq 0.25$. Em redes profundas, gradiente diminui exponencialmente.
- **Computacionalmente cara:** Envolve exponencial.

### 1.2 ReLU: The Game Changer

$$\text{ReLU}(z) = \max(0, z)$$

Vantagens:
- Derivada simples: $\frac{d}{dz}\text{ReLU} = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$
- Não sofre de vanishing gradient (derivada = 1 quando ativa)
- Rápido computacionalmente (só comparação)
- Permite redes mais profundas

Desvantagem:
- **Dead ReLU:** se neurônio ativa com z < 0 sempre, gradiente = 0, nunca aprende ("morto")

```python
import numpy as np
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


# Visualizar
z = np.linspace(-3, 3, 100)
a = relu(z)
da = relu_derivative(z)

fig, ax = plt.subplots()
ax.plot(z, a, label='ReLU(z)', linewidth=2)
ax.plot(z, da, label="ReLU'(z)", linewidth=2)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.legend()
ax.set_xlabel('z')
ax.set_ylabel('Ativação / Derivada')
ax.grid(True, alpha=0.3)
plt.show()
```

### 1.3 Variações de ReLU

**Leaky ReLU:**

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

Onde $\alpha \approx 0.01$. Impede "dead ReLU".

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

**ELU (Exponential Linear Unit):**

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

Suave (diferenciável até em z=0), reduz bimodal distribution de ativações.

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1, alpha * np.exp(z))
```

**GELU (Gaussian Error Linear Unit):**

$$\text{GELU}(z) = z \cdot \Phi(z)$$

Onde $\Phi(z)$ é Gaussian CDF. Usado em transformers modernos.

```python
from scipy.special import erf


def gelu(z):
    """GELU approximation."""
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))


def gelu_derivative(z):
    """GELU derivative (numerical)."""
    eps = 1e-5
    return (gelu(z + eps) - gelu(z - eps)) / (2 * eps)
```

### 1.4 Comparação Visual

```python
z = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ReLU
a_relu = relu(z)
axes[0, 0].plot(z, a_relu, label='ReLU')
axes[0, 0].set_title('ReLU')
axes[0, 0].grid(True, alpha=0.3)

# Leaky ReLU
a_lrelu = leaky_relu(z, 0.2)
axes[0, 1].plot(z, a_lrelu, label='Leaky ReLU')
axes[0, 1].set_title('Leaky ReLU (α=0.2)')
axes[0, 1].grid(True, alpha=0.3)

# ELU
a_elu = elu(z)
axes[1, 0].plot(z, a_elu, label='ELU')
axes[1, 0].set_title('ELU')
axes[1, 0].grid(True, alpha=0.3)

# GELU
a_gelu = gelu(z)
axes[1, 1].plot(z, a_gelu, label='GELU')
axes[1, 1].set_title('GELU')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Ativações Modernas')
plt.tight_layout()
plt.show()
```

## 2. Regularização: Combater Overfitting

### 2.1 O Problema: Overfitting

Model memoriza dados de treino em vez de aprender padrões gerais.

Sintomas:
- Train loss baixa, validation loss alta
- Train accuracy 99%, test accuracy 70%

### 2.2 L1 e L2 Regularization

**Ideia:** Penalizar pesos grandes.

**L2 Regularization:**

$$L_{\text{total}} = L_{\text{orig}} + \lambda \|\mathbf{W}\|_2^2 = L_{\text{orig}} + \lambda \sum w_i^2$$

**L1 Regularization:**

$$L_{\text{total}} = L_{\text{orig}} + \lambda \|\mathbf{W}\|_1 = L_{\text{orig}} + \lambda \sum |w_i|$$

**Diferença:**
- L2: Reduz todos os pesos proporcionalmente (weight decay)
- L1: Força alguns pesos para exatamente 0 (feature selection)

```python
def l2_regularization_loss(weights, lambda_reg):
    """L2 penalty."""
    return lambda_reg * np.sum([np.sum(w**2) for w in weights])


def l2_regularization_gradient(w, lambda_reg):
    """Gradient of L2 penalty."""
    return 2 * lambda_reg * w


def l1_regularization_loss(weights, lambda_reg):
    """L1 penalty."""
    return lambda_reg * np.sum([np.sum(np.abs(w)) for w in weights])


def l1_regularization_gradient(w, lambda_reg):
    """Gradient of L1 penalty (subgradient: sign(w))."""
    return lambda_reg * np.sign(w)
```

### 2.3 Dropout: Aleatoriedade que Evita Co-adaptação

**Ideia:** Durante treino, desligar neurônios aleatoriamente (com probabilidade p).

- Força rede a aprender redundância
- Durante validação/teste: usar todos os neurônios mas escalar por (1-p)

```python
def dropout_forward(a, p_keep=0.5, training=True):
    """
    a: ativações shape (batch, hidden)
    p_keep: probabilidade de manter neurônios
    """
    if not training:
        return a
    
    # Gerar máscara aleatória
    mask = np.random.binomial(1, p_keep, size=a.shape)
    # Escalar para manter expectativa
    a_dropped = a * mask / p_keep
    return a_dropped, mask


def dropout_backward(grad_a, mask, p_keep=0.5):
    """Backward pass para dropout."""
    return grad_a * mask / p_keep


# Exemplo
a = np.random.randn(32, 100)  # 32 batch, 100 neurônios
a_dropped, mask = dropout_forward(a, p_keep=0.8, training=True)
print(f"Fração desligada: {1 - np.mean(mask):.2%}")
```

### 2.4 Batch Normalization: Normaliza Ativações

**Problema:** Ativações internamente mudam de escala durante treinamento (covariate shift).

**Solução:** Normalizar cada batch internamente.

**Forward:**

$$z_{\text{norm}} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$$
$$\hat{z} = \gamma z_{\text{norm}} + \beta$$

Onde $\gamma$, $\beta$ são parâmetros treináveis ("scale" e "shift").

```python
def batch_norm_forward(z, gamma, beta, running_mean, running_var, 
                       momentum=0.9, eps=1e-5, training=True):
    """
    z: input activations shape (batch, features)
    gamma, beta: learned scale and shift
    running_mean, running_var: accumulated mean/var for inference
    """
    if training:
        # Compute batch statistics
        mu_batch = np.mean(z, axis=0, keepdims=True)
        var_batch = np.var(z, axis=0, keepdims=True)
        
        # Normalize
        z_norm = (z - mu_batch) / np.sqrt(var_batch + eps)
        
        # Update running statistics (exponential moving average)
        running_mean = momentum * running_mean + (1 - momentum) * mu_batch
        running_var = momentum * running_var + (1 - momentum) * var_batch
    else:
        # Use running statistics for inference
        z_norm = (z - running_mean) / np.sqrt(running_var + eps)
    
    # Scale and shift
    z_out = gamma * z_norm + beta
    
    return z_out, z_norm, running_mean, running_var


def batch_norm_backward(grad_out, z_norm, gamma, batch_size, eps=1e-5):
    """Backward pass for batch norm."""
    grad_gamma = np.sum(grad_out * z_norm, axis=0)
    grad_beta = np.sum(grad_out, axis=0)
    grad_z_norm = grad_out * gamma
    
    # Gradient w.r.t. normalized input
    grad_var = np.sum(grad_z_norm * z_norm * -0.5 * (1 - z_norm**2)**(- 1.5), axis=0)
    grad_mu = np.sum(grad_z_norm * -1 / np.sqrt(1 - z_norm**2 + eps), axis=0) + \
              grad_var * np.mean(-2 * z_norm, axis=0)
    
    # Gradient w.r.t. input
    grad_z = grad_z_norm / np.sqrt(1 - z_norm**2 + eps) + \
             grad_var * 2 / batch_size * z_norm + \
             grad_mu / batch_size
    
    return grad_z, grad_gamma, grad_beta
```

**Benefícios:**
- Reduz internal covariate shift
- Permite learning rates maiores
- Efeito regularizador
- Menor sensibilidade a inicialização

### 2.5 Layer Normalization

Como batch norm, mas **normaliza em cada exemplo**, não no batch.

Bom para:
- RNNs e transformers (batch size variável)
- Aplicações onde batch norm não faz sentido

```python
def layer_norm_forward(z, gamma, beta, eps=1e-5):
    """Normalize features (colunas), não batch."""
    mu = np.mean(z, axis=1, keepdims=True)
    var = np.var(z, axis=1, keepdims=True)
    z_norm = (z - mu) / np.sqrt(var + eps)
    z_out = gamma * z_norm + beta
    return z_out, z_norm
```

## 3. Combinando Tudo: Rede com Regularização

```python
class SimpleNetworkWithRegularization:
    def __init__(self, layer_sizes, activation='relu', 
                 dropout_rate=0.5, l2_lambda=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        
        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Dropout masks (stored for backward pass)
        self.dropout_masks = []
    
    def activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'elu':
            return elu(z)
        elif self.activation == 'gelu':
            return gelu(z)
        else:
            return z
    
    def activate_derivative(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'elu':
            return elu_derivative(z)
        elif self.activation == 'gelu':
            return gelu_derivative(z)
        else:
            return np.ones_like(z)
    
    def forward(self, x, training=True):
        self.activations = [x]
        self.z_values = []
        self.dropout_masks = []
        
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Linear
            z = a @ w + b
            self.z_values.append(z)
            
            # Activation  (except last layer)
            if i < len(self.weights) - 1:
                a = self.activate(z)
                
                # Dropout
                a_dropped, mask = dropout_forward(a, 1 - self.dropout_rate, training)
                if training:
                    a = a_dropped
                self.dropout_masks.append(mask)
            else:
                # Output layer: softmax
                exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            self.activations.append(a)
        
        return a
    
    def compute_total_loss(self, predictions, targets):
        """Cross-entropy + L2 regularization."""
        batch_size = len(targets)
        
        # Cross-entropy
        eps = 1e-7
        ce_loss = -np.mean(np.sum(targets * np.log(predictions + eps), axis=1))
        
        # L2 regularization
        l2_loss = l2_regularization_loss(self.weights, self.l2_lambda)
        
        return ce_loss + l2_loss


# Simples exemplo
network = SimpleNetworkWithRegularization(
    layer_sizes=[28*28, 128, 64, 10],
    activation='relu',
    dropout_rate=0.5,
    l2_lambda=0.0001
)

x_dummy = np.random.randn(32, 28*28)
y_dummy = np.eye(10)[np.random.randint(0, 10, 32)]

logits = network.forward(x_dummy, training=True)
loss = network.compute_total_loss(logits, y_dummy)
```

## 4. Checklist: Quando Usar O Quê

- **Ativação:** ReLU para maioria das cases. GELU para transformers. Evite sigmoid (problemas de convergência).
- **Dropout:** 20-50% em camadas densas. Não no output layer.
- **Batch Norm:** Excelente em CNNs. Evite em RNNs, use Layer Norm em vez.
- **L2:** Padrão. Comece com λ = 0.0001.
- **L1:** Se quiser sparsidade (zero weights).

## 5. Exercícios

### Ex. 1: Comparar Ativações

Train rede com ReLU vs Leaky ReLU vs GELU. Qual converge mais rápido?

### Ex. 2: Dropout Impact

Train com/sem dropout (mesmo learning rate). Qual generaliza melhor?

### Ex. 3: Regularization Tuning

Train com λ = 0, 0.00001, 0.0001, 0.001. Plote train vs validation loss.

## Próximo

[→ Verificação de Gradientes e Debugging](./06_verificacao_gradiente_debugging.md)
