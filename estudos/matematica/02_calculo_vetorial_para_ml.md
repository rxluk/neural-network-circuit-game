# Matemática 02: Cálculo Vetorial para ML

## Introdução: Do Escalar para o Vetor

Na escola, você aprendeu derivada de função escalarScalar $y = f(x)$:

$$
\frac{dy}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

Em ML, temos **funções multivariáveis**:

$$
L = f(\mathbf{x}) \text{ onde } \mathbf{x} \in \mathbb{R}^n \text{ (vetor de n variáveis)}
$$

Por exemplo, loss de rede neural:

$$
L = \text{MSE}(\mathbf{y}_{pred}, \mathbf{y}_{true})
$$

Onde $\mathbf{y}_{pred} = f(\mathbf{w}, \mathbf{b})$ depende de **milhões de parâmetros**.

Logo passamos de "derivada com respeito a 1 variável" para "derivadas com respeito a **cada parâmetro**".

Isso é **cálculo vetorial**.

## 1. Derivada Parcial

### 1.1 Definição

Derivada parcial de $f$ com respeito a $x_i$ é:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}
$$

Trata todas as outras variáveis como **constantes**.

### 1.2 Exemplo Concreto

$$
f(x, y) = x^2 + xy + y^2
$$

Derivada parcial em relação a $x$:

$$
\frac{\partial f}{\partial x} = 2x + y
$$

(Trata $y$ como constante, só usa a regra de potência)

Derivada parcial em relação a $y$:

$$
\frac{\partial f}{\partial y} = x + 2y
$$

```python
def f(x, y):
    return x**2 + x*y + y**2

# Em um ponto específico: (x, y) = (2, 3)
x, y = 2, 3
f_val = f(x, y)  # 4 + 6 + 9 = 19

# Aproximação numérica de derivada parcial:
h = 1e-5
df_dx = (f(x + h, y) - f(x, y)) / h  # ≈ 2*2 + 3 = 7
df_dy = (f(x, y + h) - f(x, y)) / h  # ≈ 2 + 2*3 = 8

print(f"∂f/∂x ≈ {df_dx:.6f}")  # ≈ 7
print(f"∂f/∂y ≈ {df_dy:.6f}")  # ≈ 8
```

## 2. Gradiente: O Vetor de Derivadas

### 2.1 Definição

Para função $f(\mathbf{x})$, o **gradiente** é vetor de todas as derivadas parciais:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

Shape: vetor coluna de tamanho `(n,)`.

### 2.2 Exemplo: MSE Loss

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^m (y_i^{pred} - y_i^{true})^2
$$

Onde $y_i^{pred} = \mathbf{w}^T \mathbf{x}_i + b$ (regressão linear simples).

Para um único exemplo $i$:

$$
L_i = (y_i^{pred} - y_i^{true})^2 = (\mathbf{w}^T \mathbf{x}_i + b - y_i^{true})^2
$$

Derivada em relação a $\mathbf{w}$:

$$
\frac{\partial L_i}{\partial \mathbf{w}} = 2(\mathbf{w}^T \mathbf{x}_i + b - y_i^{true}) \mathbf{x}_i
$$

```python
import numpy as np

def mse_loss(w, x, y_true):
    # x shape: (batch, features)
    # w shape: (features,)
    # y_true shape: (batch,)
    y_pred = x @ w  # (batch,)
    return np.mean((y_pred - y_true)**2)

def mse_gradient(w, x, y_true):
    # Retorna ∇L with respeito a w
    # Shape: (features,)
    y_pred = x @ w
    m = x.shape[0]
    grad_w = 2/m * x.T @ (y_pred - y_true)  # (features,)
    return grad_w

# Exemplo
x = np.random.randn(32, 5)  # 32 exemplos, 5 features
w = np.random.randn(5)
y_true = np.random.randn(32)

loss = mse_loss(w, x, y_true)
grad = mse_gradient(w, x, y_true)

print(f"Loss: {loss:.4f}")
print(f"Gradient shape: {grad.shape}")  # (5,)
print(f"Gradient: {grad}")
```

## 3. Interpretação Geométrica do Gradiente

### 3.1 Direção de Máxima Ascensão

Gradiente aponta na **direção de maior aumento** da função.

- Se você está em ponto $\mathbf{x}$ em uma "colina" $f(\mathbf{x})$
- O gradiente $\nabla f$ mostra "para onde subir mais rápido"

Para **descida do gradiente** (minimização), você vai no **sentido oposto**:

$$
\mathbf{w}_{novo} = \mathbf{w}_{antigo} - \alpha \nabla L
$$

Onde $\alpha$ = learning rate (tamanho do passo).

```python
# Minimização por gradient descent
learning_rate = 0.01
w = np.zeros(5)

for epoch in range(100):
    loss = mse_loss(w, x, y_true)
    grad = mse_gradient(w, x, y_true)
    
    # Passo de descida
    w = w - learning_rate * grad
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### 3.2 Magnitude do Gradiente

$\|\nabla f\|$ = rapidez da mudança.

- Se $\|\nabla f\| \approx 0$ → estamos perto de **mínimo local** ✓
- Se $\|\nabla f\|$ muito grande → curvatura é acentuada

## 4. Matriz Jacobiana

### 4.1 Definição

Se função leva **vetor a vetor**:

$$
\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m, \quad \mathbf{y} = \mathbf{f}(\mathbf{x})
$$

Matriz Jacobiana é:

$$
\mathbf{J} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix}
$$

Shape: `(m, n)`.

### 4.2 Exemplo: Forward Pass de Rede

```
Entrada: x = (8,)
Camada 1: y1 = σ(x @ W1 + b1), shape (14,)
Camada 2: y2 = σ(y1 @ W2 + b2), shape (2,)
```

Jacobiana de cada camada:
- $\mathbf{J}_1$ = Jacobiana da entrada para y1 = `(14, 8)`
- $\mathbf{J}_2$ = Jacobiana de y1 para y2 = `(2, 14)`

Aplicar **chain rule**:
$$
\mathbf{J}_{total} = \mathbf{J}_2 \times \mathbf{J}_1
$$

## 5. Regra da Cadeia (Chain Rule) - ESSENCIAL

### 5.1 Versão Escalar

Se $y = f(g(x))$:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

### 5.2 Versão Vetorial (mais de uma variável)

Se $L = f(\mathbf{u})$ e $\mathbf{u} = g(\mathbf{x})$:

$$
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}_{\mathbf{u}}^T \frac{\partial L}{\partial \mathbf{u}}
$$

Onde $\mathbf{J}_{\mathbf{u}}$ é Jacobiana de $\mathbf{u}$ com respeito a $\mathbf{x}$.

### 5.3 Aplicação: Backprop

Forward pass computa:

$$
\mathbf{z}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1
$$
$$
\mathbf{a}_1 = \sigma(\mathbf{z}_1)
$$
$$
\mathbf{z}_2 = \mathbf{a}_1 \mathbf{W}_2 + \mathbf{b}_2
$$
$$
\mathbf{a}_2 = \sigma(\mathbf{z}_2)
$$
$$
L = \text{MSE}(\mathbf{a}_2, \mathbf{y}_{true})
$$

Backward (chain rule):

$$
\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{a}_2} \cdot \frac{\partial \mathbf{a}_2}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{W}_2}
$$

Cada ".•" é multiplicação de matrizes (regra da cadeia).

## 6. Matriz Hessiana: Derivada Segunda

### 6.1 Definição

Para função $f(\mathbf{x})$, Hessiana é matriz de **segundas derivadas**:

$$
\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

Shape: `(n, n)` quadrada.

Simetria (teorema de Schwarz): $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.

### 6.2 Dado Geométrico: Curvatura

Hessiana mede **curvatura** da função.

- $\mathbf{H}$ positivo-definida → função "convexa" (única mínimo)
- $\mathbf{H}$ negativa-definida → função "côncava" (máximo)
- $\mathbf{H}$ indefinida → ponto de sela (não é mín nem máx)

```python
# Exemplo: Hessiana de f(x, y) = x^2 + xy + y^2
# ∂f/∂x = 2x + y
# ∂f/∂y = x + 2y

# Segundas derivadas:
# ∂²f/∂x² = 2
# ∂²f/∂x∂y = 1
# ∂²f/∂y² = 2

H = np.array([[2, 1],
              [1, 2]])
autovalores = np.linalg.eigvals(H)
print(autovalores)  # [1., 3.] todos positivos → convexa!
```

Convexidade garante gradient descent chega a **mínimo global**.

## 7. Cálculo de Derivadas com Regras Práticas

### 7.1 Regra 1: Derivada de $\mathbf{x}^T \mathbf{a}$

$$
\frac{\partial (\mathbf{x}^T \mathbf{a})}{\partial \mathbf{x}} = \mathbf{a}
$$

Prova: $\mathbf{x}^T \mathbf{a} = \sum x_i a_i$, então $\frac{\partial}{\partial x_i} = a_i$ para cada $i$.

```python
# Verificação numérica
x = np.array([1.0, 2.0, 3.0])
a = np.array([10.0, 20.0, 30.0])

def f(x):
    return np.sum(x * a)

# Gradiente numérico
grad_num = np.zeros_like(x)
h = 1e-5
for i in range(len(x)):
    x_plus = x.copy()
    x_plus[i] += h
    grad_num[i] = (f(x_plus) - f(x)) / h

print("Gradiente numérico:", grad_num)  # ≈ [10, 20, 30]
print("Gradiente analítico (regra):", a)  # [10, 20, 30]
```

### 7.2 Regra 2: Derivada de $\mathbf{x}^T \mathbf{A} \mathbf{x}$

$$
\frac{\partial (\mathbf{x}^T \mathbf{A} \mathbf{x})}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}
$$

Se $\mathbf{A}$ é simétrica: $= 2 \mathbf{A} \mathbf{x}$.

### 7.3 Regra 3: Derivada de $\|\|\mathbf{A} \mathbf{x} - \mathbf{b}\|\|^2$

$$
\frac{\partial}{\partial \mathbf{x}} \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 = 2 \mathbf{A}^T (\mathbf{A} \mathbf{x} - \mathbf{b})
$$

Usado em Least Squares.

### 7.4 Regra 4: Derivada da Multiplicação de Matrizes

Se $\mathbf{Y} = \mathbf{X} \mathbf{W}$:

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}}
$$

Shape: `(n_in, n_out) = (n_in, m)^T @ (m, n_out)`.

## 8. Verificação Numérica de Derivadas

### 8.1 Aproximação Central

Para verificar se sua derivada analítica está correta:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)}{2h}
$$

Onde $\mathbf{e}_i$ = vetor com 1 na posição $i$, 0 no resto.

Onde $h \approx 10^{-5}$ é pequeno o suficiente para ser aproximação, mas grande o suficiente para evitar erro numérico.

```python
def numerical_gradient(f, x, h=1e-5):
    """Calcula gradiente numérico de f em x."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h_plus = x.copy()
        x_h_plus[i] += h
        x_h_minus = x.copy()
        x_h_minus[i] -= h
        grad[i] = (f(x_h_plus) - f(x_h_minus)) / (2 * h)
    return grad

def analytical_gradient_mse(w, x, y_true):
    """Seu gradiente derivado em papel."""
    y_pred = x @ w
    return 2 * x.T @ (y_pred - y_true) / len(y_true)

# Verificação
x = np.random.randn(32, 5)
w = np.random.randn(5)
y_true = np.random.randn(32)

def mse(w):
    return np.mean((x @ w - y_true)**2)

grad_numerical = numerical_gradient(mse, w)
grad_analytical = analytical_gradient_mse(w, x, y_true)

print("Diferença:", np.linalg.norm(grad_numerical - grad_analytical))
# Deve ser < 1e-7 se derivada está correta!
```

## 9. Exercícios Práticos

### Ex. 1: Calcular Gradiente em Papel

Defina $f(x, y) = 3x^2 + 2xy + y^3$.

Calcule **em papel**:
1. $\frac{\partial f}{\partial x}$
2. $\frac{\partial f}{\partial y}$
3. $\nabla f$ no ponto $(x, y) = (1, 2)$

Depois valide com NumPy usando derivada numérica.

### Ex. 2: Aplicar Chain Rule

Defina:
- $z = 2x + y$
- $u = z^2$
- $v = e^u$

Calcule $\frac{\partial v}{\partial x}$ usando chain rule passo-a-passo.

Depois implemente em NumPy e valide.

### Ex. 3: Gradient Descent Manualmente

Implementar gradient descent em MSE Loss:

```python
def gradient_descent_step(w, x, y_true, learning_rate):
    grad = analytical_gradient_mse(w, x, y_true)
    w_novo = w - learning_rate * grad
    return w_novo
```

Rode 100 passos, plote loss vs epoch, veja convergência.

## Próximo módulo

[→ Backpropagation Profundo e Detalhado](../AI/NN/03_backprop_derivadas_chain_rule.md)
