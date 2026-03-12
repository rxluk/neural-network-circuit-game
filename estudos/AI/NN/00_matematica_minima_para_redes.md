# Modulo NN 00: Matematica minima para redes neurais e simulacao

Este modulo existe para que a trilha nova seja autossuficiente. Sem ele, muita coisa de rede neural fica parecendo decoracao numerica.

## 1. Vetores

Um vetor e uma colecao ordenada de numeros.

No projeto, vetores aparecem como:

- entradas da rede
- saidas da rede
- direcao de movimento
- tangente da pista

## 2. Matrizes

Matrizes aparecem principalmente como pesos entre camadas.

Se voce tem 8 entradas e 14 neuronios ocultos, uma matriz de pesos pode ter shape:

$$
(8, 14)
$$

## 3. Produto vetor-matriz

Essa e a operacao central de uma camada neural.

$$
y = xW
$$

Ela mistura as entradas com os pesos e produz novos sinais.

## 4. Soma ponderada

Cada neuronio faz algo conceitualmente assim:

$$
z = x_1w_1 + x_2w_2 + ... + x_nw_n + b
$$

## 5. Bias

Bias e um deslocamento adicional. Ele ajuda o neuronio a ativar de forma mais flexivel.

## 6. Ativacao sigmoid

No projeto:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Ela comprime valores para a faixa entre 0 e 1.

## 7. Normalizacao

Os sensores e a velocidade sao normalizados para escalas mais controladas. Isso ajuda a rede a operar melhor numericamente.

## 8. Distancia em 2D

Distancias sao essenciais para sensores, pista e progresso.

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

## 9. Angulo, seno e cosseno

Para mover o carro no plano:

$$
x = x + v\cos(\theta)
$$

$$
y = y + v\sin(\theta)
$$

## 10. Produto escalar

Ajuda a comparar direcoes. E por isso e util para detectar contramao.

## 11. O que voce precisa dominar de verdade

- shape de vetor e matriz
- produto `x @ W + b`
- nocao de ativacao
- trigonometria basica para movimento

## Exercicios

### Exercicio 1

Calcule no papel uma camada simples com 3 entradas, 2 neuronios, uma matriz de pesos e um bias.

### Exercicio 2

Implemente um script que receba angulo e velocidade e atualize uma posicao 2D por 30 passos.
