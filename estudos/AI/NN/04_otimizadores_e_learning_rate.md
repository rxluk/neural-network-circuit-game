# Redes Neurais 04: Otimizadores - SGD, Momentum, RMSprop, Adam

## Introdução: Por Que Otimizadores Importam

Backprop te dá o **gradiente**.

Mas gradiente sozinho não é suficiente. Você precisa de **estratégia para dar passos inteligentes** na direção da descida.

Diferença entre otimizadores:

- **SGD puro:** Passo "burro", mesmo tamanho sempre
- **Momentum:** "Memória" da direção anterior, suaviza oscilações
- **RMSprop:** Adapta tamanho do passo por dimensão
- **Adam:** Combina momentum + adaptive learning rate ← **Modern Standard**

Objetivo deste módulo:

1. Entender **por que** cada otimizador foi criado
2. Ver **formalmente** as atualções
3. Implementar em NumPy
4. Comparar empiricamente em toy problem

## 1. Baseline: Gradient Descent Vanilla

### 1.1 O Algoritmo Mais Simples

```
Para cada epoch:
  Para cada batch:
    Calcular gradiente ∇L
    Atualizar: θ ← θ - α ∇L
```

### 1.2 Problema 1: Learning Rate Fixo

Se learning rate $\alpha$ é muito pequeño → convergência lenta.
Se muito grande → overshoots, diverge.

```python
import numpy as np
import matplotlib.pyplot as plt


def sphere_2d(x1, x2):
    """
    Função convexa simples: f(x) = x1^2 + x2^2
    Mínimo em (0, 0) com valor 0.
    Gradiente: ∇f = [2x1, 2x2]
    """
    return x1**2 + x2**2


def sphere_gradient(x):
    return 2 * x


# SGD vanilla
def sgd_vanilla(x_init, learning_rate, iterations):
    trajectory = [x_init.copy()]
    losses = [sphere_2d(x_init[0], x_init[1])]
    x = x_init.copy()
    
    for _ in range(iterations):
        grad = sphere_gradient(x)
        x = x - learning_rate * grad
        
        trajectory.append(x.copy())
        losses.append(sphere_2d(x[0], x[1]))
    
    return x, np.array(trajectory), np.array(losses)


# Teste com diferentes learning rates
x_init = np.array([5.0, 5.0])
_, traj_slow, loss_slow = sgd_vanilla(x_init, 0.01, 100)
_, traj_fast, loss_fast = sgd_vanilla(x_init, 0.2, 100)
_, traj_diverge, loss_diverge = sgd_vanilla(x_init, 0.5, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Loss over time
axes[0].plot(loss_slow, label='α=0.01 (lento)')
axes[0].plot(loss_fast, label='α=0.2 (bom)')
axes[0].plot(loss_diverge, label='α=0.5 (diverge)')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Iteração')
axes[0].legend()
axes[0].set_yscale('log')

# Plot 2: 2D trajectory
axes[1].plot(traj_slow[:, 0], traj_slow[:, 1], 'o-', label='α=0.01', markersize=3)
axes[1].plot(traj_fast[:, 0], traj_fast[:, 1], 'o-', label='α=0.2', markersize=3)
axes[1].plot(traj_diverge[:, 0], traj_diverge[:, 1], 'o-', label='α=0.5', markersize=3)
axes[1].plot([0], [0], 'r*', markersize=15, label='Mínimo')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Problema Visual:** α fixo é difícil de escolher. E em ML real com 1M dimensões, cada dimensão pode ter "scales" diferentes.

## 2. Momentum: Adiciona "Inércia"

### 2.1 Intuição Física

Pense em bola rolando morro abaixo:

- Bola não apenas segue gradiente instantâneo
- Tem **velocidade acumulada** (momentum)
- Se descida muda de direção, bola não muda instantaneamente
- Reduz oscilações

### 2.2 Formulação Matemática

Mantém vetor de velocidade $\mathbf{v}$:

$$\mathbf{v} \leftarrow \beta \mathbf{v} + (1 - \beta) \nabla L$$
$$\theta \leftarrow \theta - \alpha \mathbf{v}$$

Onde $\beta \in [0.9, 0.99]$ é "coeficiente de fricção".

### 2.3 Versão "Clássica" (Nesterov)

Ligeiramente diferente, costuma convergir mais rápido:

$$\mathbf{v} \leftarrow \beta \mathbf{v} - \alpha \nabla L$$
$$\theta \leftarrow \theta + \mathbf{v}$$

### 2.4 Implementação

```python
def sgd_momentum(x_init, learning_rate, beta=0.9, iterations=100):
    """SGD com Momentum."""
    trajectory = [x_init.copy()]
    losses = [sphere_2d(x_init[0], x_init[1])]
    x = x_init.copy()
    v = np.zeros_like(x)  # velocidade inicial = 0
    
    for _ in range(iterations):
        grad = sphere_gradient(x)
        v = beta * v + (1 - beta) * grad
        x = x - learning_rate * v
        
        trajectory.append(x.copy())
        losses.append(sphere_2d(x[0], x[1]))
    
    return x, np.array(trajectory), np.array(losses)


# Comparar com SGD vanilla
_, traj_vanilla, loss_vanilla = sgd_vanilla(x_init, 0.2, 100)
_, traj_momentum, loss_momentum = sgd_momentum(x_init, 0.2, 100, beta=0.9)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(loss_vanilla, 'o-', label='SGD vanilla', markersize=3)
axes[0].plot(loss_momentum, 's-', label='SGD + Momentum', markersize=3)
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Iteração')
axes[0].legend()
axes[0].set_yscale('log')

axes[1].plot(traj_vanilla[:, 0], traj_vanilla[:, 1], 'o-', label='Vanilla', markersize=3)
axes[1].plot(traj_momentum[:, 0], traj_momentum[:, 1], 's-', label='Momentum', markersize=3)
axes[1].plot([0], [0], 'r*', markersize=15)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Observação:** Momentum converge mais rápido e com menos oscilação.

## 3. RMSprop: Adaptive Learning Rate per Dimension

### 3.1 Problema: Escalas Diferentes por Dimensão

Em rede neural, algumas direções têm "curvatura" muito maior:

- Direção 1: curvatura pequena → needs grande learning rate
- Direção 2: curvatura grande → needs pequeno learning rate

RMSprop adapta learning rate em **cada dimensão** baseado em **histórico de gradientes**.

### 3.2 Formulação

Manter segundo momento (quadrado) do gradiente:

$$\mathbf{s} \leftarrow \rho \mathbf{s} + (1 - \rho) (\nabla L)^2$$
$$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\mathbf{s} + \epsilon}} \odot \nabla L$$

Onde:
- $\mathbf{s}$ = accumulated squared gradient
- $\odot$ = element-wise division
- $\epsilon = 10^{-8}$ evita divisão por zero
- $\rho \approx 0.9$ decay rate

**Intuição:** Se gradiente em dimensão $i$ historicamente grande → $s_i$ fica grande → divide por $\sqrt{s_i}$ grande → learning rate fica pequeno.

### 3.3 Implementação

```python
def rmsprop(x_init, learning_rate, rho=0.9, epsilon=1e-8, iterations=100):
    """RMSprop optimizer."""
    trajectory = [x_init.copy()]
    losses = [sphere_2d(x_init[0], x_init[1])]
    x = x_init.copy()
    s = np.zeros_like(x)  # accumulated squared gradient
    
    for _ in range(iterations):
        grad = sphere_gradient(x)
        s = rho * s + (1 - rho) * (grad ** 2)
        x = x - learning_rate * grad / (np.sqrt(s) + epsilon)
        
        trajectory.append(x.copy())
        losses.append(sphere_2d(x[0], x[1]))
    
    return x, np.array(trajectory), np.array(losses)


# Comparar
_, traj_rmsprop, loss_rmsprop = rmsprop(x_init, 0.1, iterations=100)

fig, ax = plt.subplots()
ax.plot(loss_vanilla, 'o-', label='SGD vanilla', markersize=3)
ax.plot(loss_momentum, 's-', label='Momentum', markersize=3)
ax.plot(loss_rmsprop, '^-', label='RMSprop', markersize=3)
ax.set_ylabel('Loss')
ax.set_xlabel('Iteração')
ax.legend()
ax.set_yscale('log')
plt.show()
```

## 4. Adam: Combinando Momentum + Adaptive Learning Rate

### 4.1 A Revolução: Adam

Adam = **Ada**ptive **M**oment estimation.

Combina:
- **Primeiro momento (mean):** como momentum
- **Segundo momento (variance):** como RMSprop

Resultado: **Gold standard de otimizadores modernos.**

### 4.2 Formulação

Manter dois momentos:

$$\mathbf{m} \leftarrow \beta_1 \mathbf{m} + (1 - \beta_1) \nabla L \quad \text{(primeiro momento)}$$
$$\mathbf{v} \leftarrow \beta_2 \mathbf{v} + (1 - \beta_2) (\nabla L)^2 \quad \text{(segundo momento)}$$

Bias correction (importante no início):

$$\hat{\mathbf{m}} \leftarrow \frac{\mathbf{m}}{1 - \beta_1^t}, \quad \hat{\mathbf{v}} \leftarrow \frac{\mathbf{v}}{1 - \beta_2^t}$$

Update:

$$\theta \leftarrow \theta - \alpha \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}$$

Parâmetros padrão:
- $\beta_1 = 0.9$ (momentum coeficiente)
- $\beta_2 = 0.999$ (RMSprop coeficiente)
- $\alpha = 0.001$ (learning rate)
- $\epsilon = 10^{-8}$

### 4.3 Implementação Completa

```python
def adam(x_init, learning_rate=0.001, beta1=0.9, beta2=0.999, 
         epsilon=1e-8, iterations=100):
    """Adam optimizer."""
    trajectory = [x_init.copy()]
    losses = [sphere_2d(x_init[0], x_init[1])]
    x = x_init.copy()
    m = np.zeros_like(x)  # primeiro momento
    v = np.zeros_like(x)  # segundo momento
    
    for t in range(1, iterations + 1):
        grad = sphere_gradient(x)
        
        # Update biased momentos
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parâmetro
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        trajectory.append(x.copy())
        losses.append(sphere_2d(x[0], x[1]))
    
    return x, np.array(trajectory), np.array(losses)


# Comparar tudo
_, traj_adam, loss_adam = adam(x_init, iterations=100)

fig, ax = plt.subplots()
ax.plot(loss_vanilla, 'o-', label='SGD vanilla', markersize=2)
ax.plot(loss_momentum, 's-', label='Momentum', markersize=2)
ax.plot(loss_rmsprop, '^-', label='RMSprop', markersize=2)
ax.plot(loss_adam, 'd-', label='Adam', markersize=2)
ax.set_ylabel('Loss')
ax.set_xlabel('Iteração')
ax.legend()
ax.set_yscale('log')
plt.title('Comparação de Otimizadores em Esfera (f=x1²+x2²)')
plt.show()
```

## 5. Learning Rate Scheduling

### 5.1 Problema: Learning Rate Fixo

Convergência rápida no início com learning rate alto.
Mas depois **overshoots ao chegar perto do mínimo**.

Solução: **Reduzir learning rate com o tempo**.

### 5.2 Estratégias Comuns

**Step Decay:**

```
α(epoch) = α_0 × decay_rate^(floor(epoch / step_size))
```

```python
def step_decay_schedule(epoch, initial_lr, decay_rate=0.5, step_size=10):
    return initial_lr * (decay_rate ** (epoch // step_size))
```

**Exponential Decay:**

```
α(epoch) = α_0 × e^(-decay_rate × epoch)
```

```python
def exponential_decay_schedule(epoch, initial_lr, decay_rate=0.01):
    return initial_lr * np.exp(-decay_rate * epoch)
```

**Cosine Annealing:**

```
α(epoch) = α_min + (α_max - α_min) × (1 + cos(π × epoch / total_epochs)) / 2
```

```python
def cosine_annealing_schedule(epoch, total_epochs, alpha_min=1e-4, alpha_max=0.1):
    return alpha_min + (alpha_max - alpha_min) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
```

### 5.3 Exemplo com Learning Rate Schedule

```python
def adam_with_schedule(x_init, schedule_fn, beta1=0.9, beta2=0.999,  
                       epsilon=1e-8, iterations=100):
    """Adam with learning rate schedule."""
    trajectory = [x_init.copy()]
    losses = [sphere_2d(x_init[0], x_init[1])]
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for t in range(1, iterations + 1):
        grad = sphere_gradient(x)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Learning rate adaptado pelo schedule
        lr = schedule_fn(t - 1, iterations)
        
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        trajectory.append(x.copy())
        losses.append(sphere_2d(x[0], x[1]))
    
    return x, np.array(trajectory), np.array(losses)


# Teste schedules
_, _, loss_fixed = adam(x_init, learning_rate=0.1, iterations=100)
_, _, loss_step = adam_with_schedule(x_init,
    lambda e, t: step_decay_schedule(e, 0.1), iterations=100)
_, _, loss_cosine = adam_with_schedule(x_init,
    lambda e, t: cosine_annealing_schedule(e, t), iterations=100)

fig, ax = plt.subplots()
ax.plot(loss_fixed, label='Adam fixed α=0.1')
ax.plot(loss_step, label='Adam + Step Decay')
ax.plot(loss_cosine, label='Adam + Cosine Annealing')
ax.set_ylabel('Loss')
ax.set_xlabel('Época')
ax.legend()
ax.set_yscale('log')
plt.show()
```

## 6. Comparação Prática: Real Neural Network

```python
# Setup: Rede simples em Fashion-MNIST
# (Dataset maior, ~10x vezes mais realista)

def create_simple_network():
    """Rede 784 → 128 → 10."""
    W1 = np.random.randn(784, 128) * 0.01
    b1 = np.zeros(128)
    W2 = np.random.randn(128, 10) * 0.01
    b2 = np.zeros(10)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward_pass(x, params):
    """Forward: 784 → 128 (ReLU) → 10 (softmax)."""
    z1 = x @ params['W1'] + params['b1']
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ params['W2'] + params['b2']
    # Softmax
    exp_logits = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return a1, a2, z1, z2


def compute_loss(a2, y_onehot):
    """Cross-entropy loss."""
    eps = 1e-7
    return -np.mean(np.sum(y_onehot * np.log(a2 + eps), axis=1))


# Observação: implementação completa de forward/backward seria longo.
# Aqui apenas mostramos a ideia de comparação em problema real.
# Na prática, você usaria bibliotecas (PyTorch, TensorFlow).
```

## 7. Comparação Tabular

| Otimizador | Momentum? | Adaptive LR? | Vantagens | Desvantagens |
|-----------|-----------|-----------|-----------|-----------|
| SGD Vanilla | ✗ | ✗ | Simples | Sensível a learning rate |
| SGD + Momentum | ✓ | ✗ | Reduz oscilação | Ainda requer tuning |
| RMSprop | ✗ | ✓ | Per-dimension LR | Sem momentum |
| **Adam** | ✓ | ✓ | Gold standard, robusto | Requer menos tuning |

**Recomendação Prática:**
- Comece com **Adam** (learning rate = 0.001)
- Se não convergir bem, experimente **learning rate schedule**
- Para problemas em que convergência é crítica, considere SGD com momentum depois de vias Adam

## 8. Exercícios

### Ex. 1: Implementar do Zero

Implemente SGD vanilla, Momentum, RMSprop, Adam **totalmente do zero**.

Teste em função menor (não neural network).

### Ex. 2: Tuning em Toy Network

Implemente rede simples (~3 camadas, ~100 neurônios).

Train com cada otimizador, compare velocidade convergência e loss final.

### Ex. 3: Learning Rate Schedule Impact

Train mesma rede com:
- Fixo α = 0.001
- Step decay
- Cosine annealing

Plote loss vs epoch. Qual converge melhor?

## Próximo

[→ Ativações e Regularização](./05_ativacoes_modernas_regularizacao.md)
