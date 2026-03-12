# Gabarito 02: NumPy e matematica

1. Um vetor `x = np.array([...])` deve ter `shape` como `(4,)` se tiver 4 entradas.
2. Uma matriz `W` com shape `(4, 3)` liga 4 entradas a 3 neuronios. Um bias `b` com shape `(3,)` complementa essa camada.
3. O resultado de `x @ W + b` deve ter shape `(3,)`, porque produz uma saida por neuronio da camada de destino.
4. A sigmoid correta em NumPy segue a forma `1 / (1 + np.exp(-x))`, idealmente com `np.clip` para robustez numerica em casos maiores.
5. O movimento 2D deve usar `x += v * cos(theta)` e `y += v * sin(theta)`.
6. Normalizar sensores e velocidade significa trazer grandezas para escalas comparaveis, facilitando o comportamento numerico da rede.
7. `np.stack([x, x, x, x, x])` cria um batch com shape `(5, 4)` se `x` tiver 4 entradas.
8. No projeto, `np.einsum` faz um forward em lote para varias redes e varios agentes ao mesmo tempo.
