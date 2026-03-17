# Matemática 01: Álgebra Linear Profunda para ML

## Introdução: Por que álgebra linear?

ML é basicamente **transformação de vetores**.

- Dados: `(3000, 784)` = 3000 imagens de 784 pixels cada
- Rede: aplica transformação linear (matriz)
- Resultado: `(3000, 10)` = 3000 probabilidades de 10 classes

Tudo que acontece é **multiplicação de matrizes + não-linearidades**.

Se você não domina álgebra linear, está sabotando a si mesmo.

## 1. Revisão: Vetores

### 1.1 O que é vetor?

Vetor é **lista ordenada de números** com **direção e magnitude**.

```python
import numpy as np

v = np.array([1.0, 2.0, 3.0])  # vetor em 3D
print(v.shape)  # (3,)
```

Geometricamente:

```
    ↗️ (1, 2, 3)
   /
  / = seta do (0,0,0) até (1,2,3)
 /
O (0, 0, 0)
```

### 1.2 Operações básicas

**Soma (elemento-wise):**

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ u_3 + v_3 \end{bmatrix}
$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
resultado = u + v  # [5, 7, 9]
```

**Multiplicação por escalar:**

$$
\alpha \mathbf{v} = \alpha \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} \alpha v_1 \\ \alpha v_2 \\ \alpha v_3 \end{bmatrix}
$$

```python
escalar = 2.0
v = np.array([1, 2, 3])
resultado = escalar * v  # [2, 4, 6]
```

## 2. Norms (Normas): Medindo Tamanho

### 2.1 Norma L2 (Euclidiana)

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}
$$

É o **comprimento** do vetor (teorema de Pitágoras em n dimensões).

Exemplo concreto:

$$
\|\mathbf{v}\|_2 = \|(3, 4)^T\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

```python
v = np.array([3.0, 4.0])
norma_l2 = np.linalg.norm(v, ord=2)
print(norma_l2)  # 5.0

# Ou manual:
norma_manual = np.sqrt(np.sum(v**2))
print(norma_manual)  # 5.0
```

### 2.2 Norma L1 (Manhattan)

$$
\|\mathbf{v}\|_1 = |v_1| + |v_2| + ... + |v_n|
$$

Soma dos valores absolutos. Menos sensível a outliers.

```python
v = np.array([3.0, 4.0])
norma_l1 = np.linalg.norm(v, ord=1)
print(norma_l1)  # 7.0
```

### 2.3 Quando usar cada uma

- **L2:** Padrão em ML, deep learning. Suave, diferenciável.
- **L1:** Regularização sparse (força alguns pesos para 0). Robusta.

**ARMADILHA:** Normalizar entrada por L2 muda o modelo!

```python
v = np.array([1.0, 2.0, 3.0])
v_normalizado = v / np.linalg.norm(v)  # divide por 5.477
print(v_normalizado)  # ~[0.182, 0.365, 0.547]
```

## 3. Produto Interno (Dot Product)

### 3.1 Definição

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = u_1 v_1 + u_2 v_2 + ... + u_n v_n
$$

Resultado é **escalar**.

Exemplo:

$$
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}^T \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32
$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot = np.dot(u, v)  # 32
# Ou:
dot = u @ v  # 32
```

### 3.2 Interpretação geométrica

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Onde $\theta$ é o ângulo entre os vetores.

- Se vetores apontam "mesma direção" → $\cos(\theta) \approx 1$ → dot alto
- Se perpendiculares → $\cos(\theta) = 0$ → dot = 0
- Se opostos → $\cos(\theta) = -1$ → dot negativo

```python
# Mesmo sentido
u = np.array([1, 0])
v = np.array([2, 0])
dot = u @ v  # 2.0
# Ângulo = 0°, cos(0) = 1, |u||v|cos = 1 * 2 * 1 = 2 ✓

# Perpendicular
u = np.array([1, 0])
v = np.array([0, 1])
dot = u @ v  # 0.0
# Ângulo = 90°, cos(90) = 0 ✓

# Oposto
u = np.array([1, 0])
v = np.array([-1, 0])
dot = u @ v  # -1.0
# Ângulo = 180°, cos(180) = -1 ✓
```

### 3.3 Uso em ML: Similaridade

Em redes neurais, a camada `x @ W + b` é basicamente:

- `x` = vetor de 8 dimensões (sensores)
- `W` = matriz de 8×14 = 14 "detectores"
- Cada linha de `W` é um vetor de 8 dimensões
- `x @ W` = dot product de `x` com cada linha de `W`

Se `x` é similar a uma linha de `W` → dot product alto → ativação forte naquela unidade.

## 4. Matrizes: Extensão para 2D

### 4.1 O que é matriz

Matriz é **tabela de números** arranjo 2D.

$$
\mathbf{M} = \begin{bmatrix} m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23} \\ m_{31} & m_{32} & m_{33} \end{bmatrix}
$$

Shape: `(3, 3)` = 3 linhas, 3 colunas.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(M.shape)  # (3, 3)
print(M[0, 1])  # 2 (linha 0, coluna 1)
```

### 4.2 Transposição

$$
\mathbf{M}^T = \text{troca linhas ↔ colunas}
$$

$$
\mathbf{M} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \mathbf{M}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
$$

```python
M = np.array([[1, 2],
              [3, 4]])
M_T = M.T
print(M_T)
# [[1 3]
#  [2 4]]
```

### 4.3 Multiplicação Matriz × Vetor

$$
\mathbf{M} \mathbf{v} = \begin{bmatrix} m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23} \\ m_{31} & m_{32} & m_{33} \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} m_{11}v_1 + m_{12}v_2 + m_{13}v_3 \\ m_{21}v_1 + m_{22}v_2 + m_{23}v_3 \\ m_{31}v_1 + m_{32}v_2 + m_{33}v_3 \end{bmatrix}
$$

Cada linea da matriz faz dot product com o vetor.

Shapes: `(m, n) @ (n,) → (m,)`

```python
M = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # (3, 2)
v = np.array([10, 20])  # (2,)
resultado = M @ v  # (3,)
# resultado[0] = 1*10 + 2*20 = 50
# resultado[1] = 3*10 + 4*20 = 110
# resultado[2] = 5*10 + 6*20 = 170
print(resultado)  # [50, 110, 170]
```

### 4.4 Multiplicação Matriz × Matriz

$$
\mathbf{A} \mathbf{B} \text{ onde } \mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p} \rightarrow \mathbf{C} \in \mathbb{R}^{m \times p}
$$

Cada elemento $c_{ij}$ é o dot product da linha $i$ de $\mathbf{A}$ com coluna $j$ de $\mathbf{B}$.

```python
A = np.array([[1, 2],
              [3, 4]])  # (2, 2)
B = np.array([[5, 6, 7],
              [8, 9, 10]])  # (2, 3)
C = A @ B  # (2, 3)
# C[0, 0] = 1*5 + 2*8 = 21
# C[0, 1] = 1*6 + 2*9 = 24
# C[0, 2] = 1*7 + 2*10 = 27
# C[1, 0] = 3*5 + 4*8 = 47
# etc
print(C)
# [[21 24 27]
#  [47 54 61]]
```

**CRÍTICO:** Shapes precisam casar!

```
(m, n) @ (n, p) = (m, p) ✓
(m, n) @ (p, n) = ERRO ✗
```

### 4.5 Broadcasting (quando shapes não casam exato)

NumPy é *inteligente* e expande dimensões quando faz sentido.

```python
A = np.random.randn(32, 8)  # batch de 32, features 8
B = np.random.randn(8, 14)  # 8 → 14 (camada densa)
b = np.random.randn(14)     # viés

# Sem broadcasting, precisaríamos:
# resultado = np.ones((32, 1)) @ b.reshape(1, 14)

# Com broadcasting:
resultado = A @ B + b  # (32, 14) + (14,) = (32, 14)
# Python expande (14,) para (1, 14) automaticamente
```

Regras de broadcasting (direita para esquerda):
1. Dimensões iguais → OK
2. Uma dimensão = 1 → expande
3. Falta dimensão → adiciona 1 à esquerda → expande

## 5. Identidade, Inversa, Determinante

### 5.1 Matriz Identidade

$$
\mathbf{I} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

Neutro da multiplicação: $\mathbf{M} \mathbf{I} = \mathbf{M}$.

```python
I = np.eye(3)
M = np.random.randn(3, 3)
resultado = M @ I
print(np.allclose(resultado, M))  # True
```

### 5.2 Inversa

Se $\mathbf{M}$ é quadrada e invertível:

$$
\mathbf{M} \mathbf{M}^{-1} = \mathbf{I}
$$

```python
M = np.array([[1, 2],
              [3, 4]], dtype=float)
M_inv = np.linalg.inv(M)
identidade = M @ M_inv
print(identidade)
# [[1. 0.]
#  [0. 1.]]
```

**ARMADILHA:** Nem toda matriz é invertível!

```python
M_singular = np.array([[1, 2],
                       [2, 4]])  # linha 2 = 2 × linha 1
M_inv = np.linalg.inv(M_singular)  # ERRO!
```

### 5.3 Determinante

Número escalar que mede:
- Se matriz é invertível (det ≠ 0 → invertível)
- "Fator de escala" da transformação

$$
\det(\mathbf{M}) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc
$$

```python
M = np.array([[1, 2],
              [3, 4]], dtype=float)
det = np.linalg.det(M)
print(det)  # -2.0

# Verificar invertibilidade:
if abs(det) > 1e-10:
    M_inv = np.linalg.inv(M)  # OK
else:
    print("Matriz singular, não invertível")
```

## 6. Autovalores e Autovetores

### 6.1 Definição

Para matriz $\mathbf{M}$ quadrada, autopar $(\lambda, \mathbf{v})$ satisfaz:

$$
\mathbf{M} \mathbf{v} = \lambda \mathbf{v}
$$

Onde:
- $\lambda$ = autovalor (escalar)
- $\mathbf{v}$ = autovetor (vetor não-nulo)

Interpretação: multiplicar $\mathbf{M}$ por $\mathbf{v}$ só **escala** $\mathbf{v}$, não muda direção.

### 6.2 Exemplo concreto

$$
\mathbf{M} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

Encontrar $(\lambda, \mathbf{v})$ tal que $\mathbf{M} \mathbf{v} = \lambda \mathbf{v}$.

Resolvendo:

$$
\det(\mathbf{M} - \lambda \mathbf{I}) = 0
$$

$$
\det\begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix} = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0
$$

$$
(\lambda - 1)(\lambda - 3) = 0 \rightarrow \lambda_1 = 1, \lambda_2 = 3
$$

```python
M = np.array([[2, 1],
              [1, 2]], dtype=float)
autovalores, autovetores = np.linalg.eig(M)
print(autovalores)  # [1. 3.]
print(autovetores)  # colunas são autovetores
```

### 6.3 Uso em ML

**Exemplo:** Análise de Componentes Principais (PCA)

Se covarância dos dados é $\mathbf{\Sigma}$, autovetores de $\mathbf{\Sigma}$ são **direções principais** dos dados.

Autovalores = **variância** naquela direção.

Ordene autovalores (maior primeiro) → principais direções de variação.

## 7. Decomposições Importantes

### 7.1 Decomposição Espectral (Eigendecomposition)

Para matriz **simétrica** $\mathbf{M}$:

$$
\mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T
$$

Onde:
- $\mathbf{Q}$ = matriz dos autovetores (colunas)
- $\mathbf{\Lambda}$ = matriz diagonal com autovalores

```python
M = np.array([[4, 2],
              [2, 3]], dtype=float)  # simétrica
autovalores, Q = np.linalg.eig(M)
Lambda = np.diag(autovalores)

# Verificar: M = Q @ Lambda @ Q.T
M_reconstruida = Q @ Lambda @ Q.T
print(np.allclose(M, M_reconstruida))  # True
```

### 7.2 Decomposição em Valores Singulares (SVD)

Para **qualquer** matriz $\mathbf{M}$ (não precisa ser quadrada):

$$
\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

Onde:
- $\mathbf{U}$ = matriz ortogonal `(m, m)`
- $\mathbf{\Sigma}$ = diagonal `(m, n)`, entradas ≥ 0
- $\mathbf{V}$ = matriz ortogonal `(n, n)`

```python
M = np.random.randn(3, 5)
U, sigma, VT = np.linalg.svd(M)
# M ≈ U @ np.diag(sigma) @ VT
```

**Significado:** Decompõe transformação em:
1. Rotação ($\mathbf{V}^T$)
2. Escala ($\mathbf{\Sigma}$)
3. Rotação ($\mathbf{U}$)

Usado em PCA, Least Squares, compressão.

## 8. Aplicação Prática: Problema Ax = b

Um dos problemas mais comuns em ML:

$$
\mathbf{A} \mathbf{x} = \mathbf{b}
$$

Dado $\mathbf{A}$ e $\mathbf{b}$, encontrar $\mathbf{x}$.

### 8.1 Caso 1: Solução única (A quadrada, invertível)

$$
\mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
$$

```python
A = np.array([[2, 1],
              [1, 3]], dtype=float)
b = np.array([8, 9], dtype=float)
x = np.linalg.solve(A, b)  # mais estável que inv(A) @ b
print(x)  # [3. 2.]
```

### 8.2 Caso 2: Overdetermined (mais linhas que colunas)

Não há solução exata. Encontrar $\mathbf{x}$ que minimize $\|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2$ (Least Squares).

$$
\mathbf{x} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}
$$

Ou usando SVD (mais estável):

```python
A = np.random.randn(10, 5)  # 10 equações, 5 incógnitas
b = np.random.randn(10)
x = np.linalg.lstsq(A, b, rcond=None)[0]
```

## 9. Verificação de Conceitos: Exercícios

### Ex. 1: Cálculos à Mão

Dado:
$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Calcule **em papel** (sem computador):
1. $\mathbf{A}^T$
2. $\mathbf{A} \mathbf{B}$
3. $\|\text{coluna 1 de } \mathbf{A}\|_2$

Depois valide com NumPy.

### Ex. 2: Shapes e Broadcasting

```python
A = np.random.randn(32, 8)
B = np.random.randn(8, 14)
b = np.random.randn(14)

# Qual é o shape de cada resultado?
resultado1 = A @ B
resultado2 = A @ B + b
resultado3 = B.T @ A.T
resultado4 = (A @ B) @ B.T

# Tente prever antes de rodar!
```

### Ex. 3: Autovalores e Estabilidade

```python
M = np.array([[1, 0],
              [0, 2]], dtype=float)
autovalores, autovetores = np.linalg.eig(M)

# Interprete: o que significam os autovalores?
# Por que [[2, 0], [0, 1]] tem autovalores diferentes?
```

## Próximo módulo

[→ Cálculo Vetorial](./02_calculo_vetorial_para_ml.md)
