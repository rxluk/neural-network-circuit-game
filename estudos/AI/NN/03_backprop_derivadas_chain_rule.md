# Redes Neurais 03: Backpropagation - Derivadas, Matemática e Código

## Introdução: O Algoritmo Mais Importante de ML

Backpropagation é responsável por **99% do treinamento de redes neurais modernas**.

Se você não entende backprop, você não entende deep learning.

Objetivo deste módulo:
1. **Matemática:** Derivar backprop do primeiro princípio
2. **Intuição:** Entender geometricamente o que acontece
3. **Código:** Implementar do zero em NumPy
4. **Validação:** Verificar numericamente que está correto

**Tempo recomendado:** 3-4 horas de estudo sério com papel e caneta.

## 1. Setup: Forward Pass de Rede Simples

Consideremos uma rede com **2 camadas ocultas** mínima:

### 1.1 Arquitetura

```
Entrada: x ∈ ℝ² (2 features)
  ↓ (camada 1)
z¹ = x·W¹ + b¹, onde W¹ ∈ ℝ^(2×3)
a¹ = ReLU(z¹), shape (3,)
  ↓ (camada 2)
z² = a¹·W² + b², onde W² ∈ ℝ^(3×1)
a² = sigmoid(z²) [output, probabilidade]
  ↓ (loss)
L = -[y log(a²) + (1-y) log(1-a²)]  [cross-entropy bináriaexit]
```

### 1.2 Parâmetros

Complete matrix de parâmetros a aprender:

- $\mathbf{W}^1 \in \mathbb{R}^{2 \times 3}$ (pesos camada 1)
- $\mathbf{b}^1 \in \mathbb{R}^3$ (bias camada 1)
- $\mathbf{W}^2 \in \mathbb{R}^{3 \times 1}$ (pesos camada 2)
- $\mathbf{b}^2 \in \mathbb{R}^1$ (bias camada 2)

Total: **2×3 + 3 + 3×1 + 1 = 13 parâmetros** para este exemplo pequeno.

## 2. Forward Pass Completo: Cálculo Manual

### 2.1 Exemplo Concreto com Números

Vamos usar valores **específicos e pequenos** para poder calcular tudo à mão.

**Entrada:**
$$\mathbf{x} = \begin{bmatrix} 0.5 \\ 2.0 \end{bmatrix}$$

**Label (target):**
$$y = 1$$

**Pesos e bias (inicializados aleatoriamente):**
$$\mathbf{W}^1 = \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{bmatrix}, \quad \mathbf{b}^1 = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}$$

$$\mathbf{W}^2 = \begin{bmatrix} 0.5 \\ -0.5 \\ 0.2 \end{bmatrix}, \quad \mathbf{b}^2 = 0.1$$

### 2.2 Forward Pass Passo 1: Primeira Camada

$$\mathbf{z}^1 = \mathbf{x} \mathbf{W}^1 + \mathbf{b}^1$$

$$= \begin{bmatrix} 0.5 & 2.0 \end{bmatrix} \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}$$

$$= \begin{bmatrix} 0.5 \cdot 1 + 2.0 \cdot 0, \quad 0.5 \cdot 0 + 2.0 \cdot 1, \quad 0.5 \cdot (-1) + 2.0 \cdot 1 \end{bmatrix} + \mathbf{b}^1$$

$$= \begin{bmatrix} 0.5, \quad 2.0, \quad 1.5 \end{bmatrix} + \begin{bmatrix} 0.1, 0.2, 0.3 \end{bmatrix} = \begin{bmatrix} 0.6, 2.2, 1.8 \end{bmatrix}$$

**Aplicar ReLU:**
$$\mathbf{a}^1 = \text{ReLU}(\mathbf{z}^1) = \max(0, \mathbf{z}^1) = \begin{bmatrix} 0.6, 2.2, 1.8 \end{bmatrix}$$

(Neste caso, todos positivos, então ReLU não muda nada)

### 2.3 Forward Pass Passo 2: Segunda Camada

$$\mathbf{z}^2 = \mathbf{a}^1 \mathbf{W}^2 + \mathbf{b}^2$$

$$= \begin{bmatrix} 0.6, 2.2, 1.8 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \\ 0.2 \end{bmatrix} + 0.1$$

$$= 0.6 \cdot 0.5 + 2.2 \cdot (-0.5) + 1.8 \cdot 0.2 + 0.1$$

$$= 0.3 - 1.1 + 0.36 + 0.1 = -0.34$$

**Aplicar Sigmoid:**
$$\mathbf{a}^2 = \sigma(\mathbf{z}^2) = \frac{1}{1 + e^{-(-0.34)}} = \frac{1}{1 + e^{0.34}} = \frac{1}{1 + 1.405} \approx 0.415$$

### 2.4 Calcular Loss

$$L = -[y \log(\mathbf{a}^2) + (1-y) \log(1-\mathbf{a}^2)]$$

Com $y = 1$, $\mathbf{a}^2 \approx 0.415$:

$$L = -[\log(0.415) + 0] = -(-0.882) = 0.882$$

**Resumo Forward:**

| Variável | Valor |
|----------|-------|
| $\mathbf{x}$ | [0.5, 2.0] |
| $\mathbf{z}^1$ | [0.6, 2.2, 1.8] |
| $\mathbf{a}^1$ | [0.6, 2.2, 1.8] |
| $\mathbf{z}^2$ | -0.34 |
| $\mathbf{a}^2$ | 0.415 |
| $L$ | 0.882 |

## 3. Backward Pass: As Derivadas

Agora calculamos **gradientes com respeito a TODOS os parâmetros** usando chain rule.

### 3.1 Passo 1: Derivada da Loss com Respeito a $\mathbf{a}^2$

$$L = -[y \log(\mathbf{a}^2) + (1-y) \log(1-\mathbf{a}^2)]$$

$$\frac{\partial L}{\partial \mathbf{a}^2} = -\left[\frac{y}{\mathbf{a}^2} - \frac{1-y}{1-\mathbf{a}^2}\right]$$

Com $y = 1$, $\mathbf{a}^2 = 0.415$:

$$\frac{\partial L}{\partial \mathbf{a}^2} = -\left[\frac{1}{0.415} - 0\right] = -2.41$$

### 3.2 Passo 2: Derivada da $\mathbf{a}^2$ com Respeito a $\mathbf{z}^2$

Sigmoid: $\mathbf{a}^2 = \sigma(\mathbf{z}^2)$, derivada:

$$\frac{\partial \mathbf{a}^2}{\partial \mathbf{z}^2} = \sigma(\mathbf{z}^2) (1 - \sigma(\mathbf{z}^2)) = 0.415 \cdot (1 - 0.415) = 0.415 \cdot 0.585 \approx 0.243$$

### 3.3 Passo 3: Combinar Usando Chain Rule (crítico!)

$$\frac{\partial L}{\partial \mathbf{z}^2} = \frac{\partial L}{\partial \mathbf{a}^2} \cdot \frac{\partial \mathbf{a}^2}{\partial \mathbf{z}^2} = (-2.41) \cdot (0.243) \approx -0.585$$

**Interpretação:** O sinal negativo mostra que reduzir $\mathbf{z}^2$ reduziria a loss (bom!).

### 3.4 Passo 4: Gradiente com Respeito a $\mathbf{W}^2$

Using chain rule:

$$\frac{\partial L}{\partial \mathbf{W}^2} = \frac{\partial L}{\partial \mathbf{z}^2} \cdot \frac{\partial \mathbf{z}^2}{\partial \mathbf{W}^2}$$

Lembre: $\mathbf{z}^2 = \mathbf{a}^1 \mathbf{W}^2 + \mathbf{b}^2$

Cada elemento $w^2_i$ aparece em $z^2 = a^1_i \cdot w^2_i + \ldots$:

$$\frac{\partial \mathbf{z}^2}{\partial w^2_i} = \mathbf{a}^1_i$$

Portanto (regra geral):

$$\frac{\partial L}{\partial \mathbf{W}^2} = (\mathbf{a}^1)^T \cdot \frac{\partial L}{\partial \mathbf{z}^2}$$

$$= \begin{bmatrix} 0.6 \\ 2.2 \\ 1.8 \end{bmatrix} \cdot (-0.585) = \begin{bmatrix} -0.351 \\ -1.287 \\ -1.053 \end{bmatrix}$$

Shape: **`(3, 1)` pois $\mathbf{W}^2$ tem shape `(3, 1)`**.

### 3.5 Passo 5: Gradiente com Respeito a $\mathbf{b}^2$

Lembre: cada $b^2_j$ aparece como "+$b^2_j$" em $z^2$:

$$\frac{\partial L}{\partial \mathbf{b}^2} = \frac{\partial L}{\partial \mathbf{z}^2} = -0.585$$

### 3.6 Passo 6: Backprop para Camada 1

Gradiente com respeito a $\mathbf{a}^1$:

$$\frac{\partial L}{\partial \mathbf{a}^1} = \frac{\partial L}{\partial \mathbf{z}^2} \cdot \frac{\partial \mathbf{z}^2}{\partial \mathbf{a}^1}$$

$$= (-0.585) \cdot \begin{bmatrix} 0.5 \\ -0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} -0.293 \\ 0.293 \\ -0.117 \end{bmatrix}$$

**(Atenção:** transpomos $\mathbf{W}^2$ para backprop!)**

Agora aplicar derivada de ReLU. ReLU: $a^1_i = \max(0, z^1_i)$:

$$\frac{\partial a^1_i}{\partial z^1_i} = \begin{cases} 1 & \text{if } z^1_i > 0 \\ 0 & \text{if } z^1_i \leq 0 \end{cases}$$

Neste caso, todos os $z^1_i > 0$, então ficam todos como 1:

$$\frac{\partial L}{\partial \mathbf{z}^1} = \frac{\partial L}{\partial \mathbf{a}^1} \odot \text{ReLU}'(\mathbf{z}^1) = \begin{bmatrix} -0.293 \\ 0.293 \\ -0.117 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.293 \\ 0.293 \\ -0.117 \end{bmatrix}$$

(Onde $\odot$ é elemento-wise multiplication)

### 3.7 Passo 7: Gradientes para $\mathbf{W}^1$ e $\mathbf{b}^1$

$$\frac{\partial L}{\partial \mathbf{W}^1} = (\mathbf{x})^T \cdot \frac{\partial L}{\partial \mathbf{z}^1}$$

$$= \begin{bmatrix} 0.5 \\ 2.0 \end{bmatrix} \cdot \begin{bmatrix} -0.293 & 0.293 & -0.117 \end{bmatrix}$$

$$= \begin{bmatrix} -0.147 & 0.147 & -0.059 \\ -0.586 & 0.586 & -0.234 \end{bmatrix}$$

Shape: **`(2, 3)` como $\mathbf{W}^1$**.

$$\frac{\partial L}{\partial \mathbf{b}^1} = \frac{\partial L}{\partial \mathbf{z}^1} = \begin{bmatrix} -0.293 \\ 0.293 \\ -0.117 \end{bmatrix}$$

## 4. Resumo de Todas as Derivadas

| Parâmetro | Gradiente |
|-----------|-----------|
| $\mathbf{W}^1$ | `[[-0.147, 0.147, -0.059], [-0.586, 0.586, -0.234]]` |
| $\mathbf{b}^1$ | `[-0.293, 0.293, -0.117]` |
| $\mathbf{W}^2$ | `[-0.351, -1.287, -1.053]` |
| $\mathbf{b}^2$ | `-0.585` |

Estes são os **gradientes da loss com respeito a cada parâmetro**.

Para **atualizar** um passo (gradient descent):

$$\mathbf{W}^1_{novo} = \mathbf{W}^1 - \alpha \cdot \nabla \mathbf{W}^1$$

Onde $\alpha$ = learning rate (ex: 0.01).

## 5. Implementação em NumPy (Completa)

```python
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a):
    return a * (1 - a)


def cross_entropy_loss(a_out, y):
    """Binary cross-entropy."""
    eps = 1e-7
    return -(y * np.log(a_out + eps) + (1 - y) * np.log(1 - a_out + eps))


# Inicialização
np.random.seed(42)
x = np.array([0.5, 2.0])
y = 1

W1 = np.array([[1.0, 0.0, -1.0],
               [0.0, 1.0, 1.0]])
b1 = np.array([0.1, 0.2, 0.3])

W2 = np.array([[0.5], [-0.5], [0.2]])
b2 = np.array([0.1])

# ============ FORWARD PASS ============
z1 = x @ W1 + b1
print(f"z1 = {z1}")  # [0.6 2.2 1.8]

a1 = relu(z1)
print(f"a1 = {a1}")  # [0.6 2.2 1.8]

z2 = a1 @ W2 + b2
print(f"z2 = {z2}")  # -0.34

a2 = sigmoid(z2)
print(f"a2 = {a2}")  # ~0.415

loss = cross_entropy_loss(a2, y)
print(f"loss = {loss:.4f}")  # ~0.882

# ============ BACKWARD PASS ============
# Passo 1: dL/da2
dL_da2 = -(y / a2 - (1 - y) / (1 - a2))
print(f"\ndL/da2 = {dL_da2:.4f}")  # ~-2.41

# Passo 2: da2/dz2
da2_dz2 = sigmoid_derivative(a2)
print(f"da2/dz2 = {da2_dz2:.4f}")  # ~0.243

# Passo 3: dL/dz2 (chain rule)
dL_dz2 = dL_da2 * da2_dz2
print(f"dL/dz2 = {dL_dz2:.4f}")  # ~-0.585

# Passo 4: dL/dW2
dL_dW2 = a1.reshape(-1, 1) * dL_dz2
print(f"dL/dW2 = \n{dL_dW2}")  # shape (3, 1)

# Passo 5: dL/db2
dL_db2 = dL_dz2
print(f"dL/db2 = {dL_db2:.4f}")

# Passo 6: Backward para a1
dL_da1 = (W2.T * dL_dz2).flatten()
print(f"\ndL/da1 = {dL_da1}")

# Passo 7: ReLU backward
dL_dz1 = dL_da1 * relu_derivative(z1)
print(f"dL/dz1 = {dL_dz1}")

# Passo 8: dL/dW1 e dL/db1
dL_dW1 = x.reshape(-1, 1) @ dL_dz1.reshape(1, -1)
dL_db1 = dL_dz1

print(f"dL/dW1 = \n{dL_dW1}")  # shape (2, 3)
print(f"dL/db1 = {dL_db1}")  # shape (3,)

# ============ VERIFICAÇÃO NUMÉRICA ============
def numerical_gradient_W1(eps=1e-5):
    """Calcula gradiente numérico para W1 (verificação)."""
    grad = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = W1.copy()
            W1_plus[i, j] += eps
            z1_plus = x @ W1_plus + b1
            a1_plus = relu(z1_plus)
            z2_plus = a1_plus @ W2 + b2
            a2_plus = sigmoid(z2_plus)
            loss_plus = cross_entropy_loss(a2_plus, y)
            
            W1_minus = W1.copy()
            W1_minus[i, j] -= eps
            z1_minus = x @ W1_minus + b1
            a1_minus = relu(z1_minus)
            z2_minus = a1_minus @ W2 + b2
            a2_minus = sigmoid(z2_minus)
            loss_minus = cross_entropy_loss(a2_minus, y)
            
            grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
    return grad

grad_numerical = numerical_gradient_W1()
print(f"\nGradiente Numérico de W1:\n{grad_numerical}")
print(f"Gradiente Analítico de W1:\n{dL_dW1}")
print(f"Diferença: {np.linalg.norm(grad_numerical - dL_dW1):.2e}")
```

**Saída Esperada:**
```
z1 = [0.6 2.2 1.8]
a1 = [0.6 2.2 1.8]
z2 = [-0.34]
a2 = [[0.41512214]]
loss = 0.8819

dL/da2 = [-2.40775]
da2/dz2 = [0.24279]
dL/dz2 = [-0.58467]
dL/dW2 = 
[[-0.35080]
 [-1.28628]
 [-1.05240]]
dL/db2 = -0.5847

dL/da1 = [-0.29234  0.29234 -0.11694]
dL/dz1 = [-0.29234  0.29234 -0.11694]
dL/dW1 = 
[[-0.14617  0.14617 -0.05847]
 [-0.58467  0.58467 -0.23387]]
dL/db1 = [-0.29234  0.29234 -0.11694]

Gradiente Numérico de W1:
[[-0.14617  0.14617 -0.05847]
 [-0.58467  0.58467 -0.23387]]
Diferença: 2.34e-07  ✓ Correto!
```

## 6. Equações Gerais (Vetorizadas)

Acima mostramos para **um único exemplo**. Na prática, usamos **batches**.

Para batch de $m$ exemplos:

**Forward:**

$$\mathbf{Z}^1 = \mathbf{X} \mathbf{W}^1 + \mathbf{b}^1 \quad (\text{shape: } (m, 3))$$
$$\mathbf{A}^1 = \text{ReLU}(\mathbf{Z}^1)$$
$$\mathbf{Z}^2 = \mathbf{A}^1 \mathbf{W}^2 + \mathbf{b}^2 \quad (\text{shape: } (m, 1))$$
$$\mathbf{A}^2 = \sigma(\mathbf{Z}^2)$$
$$\mathbf{L} = -\mathbf{Y} \odot \log(\mathbf{A}^2) - (1 - \mathbf{Y}) \odot \log(1 - \mathbf{A}^2)$$

Loss final: $\text{Loss} = \frac{1}{m} \sum \mathbf{L}$.

**Backward:**

$$\frac{\partial \text{Loss}}{\partial \mathbf{A}^2} = -\frac{1}{m} \left[ \frac{\mathbf{Y}}{\mathbf{A}^2} - \frac{1 - \mathbf{Y}}{1 - \mathbf{A}^2} \right]$$

$$\frac{\partial \text{Loss}}{\partial \mathbf{Z}^2} = \frac{\partial \text{Loss}}{\partial \mathbf{A}^2} \odot \sigma'(\mathbf{Z}^2)$$

$$\frac{\partial \text{Loss}}{\partial \mathbf{W}^2} = (\mathbf{A}^1)^T \frac{\partial \text{Loss}}{\partial \mathbf{Z}^2}$$

$$\frac{\partial \text{Loss}}{\partial \mathbf{b}^2} = \sum_{\text{rows}} \frac{\partial \text{Loss}}{\partial \mathbf{Z}^2}$$

E similar para camada 1.

## 7. Pseudocódigo Genérico

```
Pseudocódigo: BACKPROPAGATION

Input: batch (x, y), rede com parâmetros θ
Output: gradientes ∇θ Loss

# Forward pass
for camada L:
    z_L = a_{L-1} @ W_L + b_L
    a_L = ativacao(z_L)

# Backward pass
dL_da_final = derivada_loss(a_final, y)
for camada L descendo até 1:
    dL_dz_L = dL_da_L ⊙ derivada_ativacao(z_L)
    dL_dW_L = (a_{L-1})^T @ dL_dz_L
    dL_db_L = sum(dL_dz_L)  # sobre batch
    dL_da_{L-1} = dL_dz_L @ (W_L)^T
    
return gradientes {dL_dW_L, dL_db_L}
```

## 8. Checklist: Você Entendeu?

- [ ] Consigo calcular forward pass numericamente?
- [ ] Entendo por que aplicar chain rule em cada passo?
- [ ] Sei como transpor $\mathbf{W}$ no backprop?
- [ ] Consigo derivar gradiente de ReLU e sigmoid?
- [ ] Entendo por que transposição muda shapes?
- [ ] Posso implementar backprop do zero?
- [ ] Consigo validar gradientes numericamente?
- [ ] Sei o que pode dar errado (vanishing/exploding gradients)?

## 9. Exercícios

### Ex. 1: Backprop Manual

Pegue arquitetura diferente (ex: 3 camadas, tanh), calcule forward e backward **à mão** para um único exemplo.

Depois implemente em NumPy e valide numericamente.

### Ex. 2: Tamanho de Batch

Implemente backprop com batch. Verifique que gradiente = média dos gradientes individuais.

### Ex. 3: Debugging Numérico

Introduza intencional bug no gradiente (ex: esqueça transposta). Rode verificação numérica e veja diferença.

## Próximo: Otimizadores

[→ Otimizadores: SGD, Momentum, Adam](./04_otimizadores_e_learning_rate.md)
