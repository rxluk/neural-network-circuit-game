# Exercicios 02: NumPy e matematica

## Objetivo

Fixar vetores, matrizes, shapes e operacoes fundamentais da rede.

## Lista

1. Crie um vetor `x` com 4 entradas e imprima seu `shape`.
2. Crie uma matriz `W` com shape `(4, 3)` e um bias `b` com shape `(3,)`.
3. Calcule `x @ W + b` e explique o shape do resultado.
4. Implemente uma funcao `sigmoid(x)` com NumPy.
5. Dado um angulo e uma velocidade, atualize `x` e `y` por 10 passos usando `cos` e `sin`.
6. Explique com suas palavras o que significa normalizar sensores e velocidade.
7. Use `np.stack` para empilhar 5 vetores de entrada iguais e forme um batch.
8. Pesquise no proprio codigo do projeto um uso de `np.einsum` e explique o que ele esta fazendo.

## Desafio extra

Implemente uma camada neural completa com `x @ W + b` e ativacao sigmoid, depois rode para 20 entradas aleatorias.
