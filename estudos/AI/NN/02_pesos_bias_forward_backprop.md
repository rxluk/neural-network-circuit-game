# Modulo NN 02: Pesos, bias, forward e backprop

Este modulo conecta o essencial.

## 1. Pesos

Pesos controlam influencia entre sinais.

Se um peso cresce em valor absoluto, aquela conexao ganha impacto maior.

## 2. Bias

Bias desloca a ativacao do neuronio.

Ele ajuda a rede a representar comportamentos mais flexiveis.

## 3. Forward pass

Forward pass e o fluxo entrada -> camada oculta -> saida.

No projeto:

$$
a_1 = \sigma(xW_1 + b_1)
$$

$$
a_2 = \sigma(a_1W_2 + b_2)
$$

## 4. Ativacao sigmoid

Usada no projeto:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

## 5. Como a saida vira acao

- primeira saida: remapeada para `[-1, 1]`
- segunda saida: mantida em `[0, 1]`

## 6. O que e backpropagation

Backpropagation e o metodo que calcula como cada peso deve mudar para reduzir o erro, usando gradientes.

## 7. O projeto usa backprop?

Nao. E muito importante guardar isso.

O projeto usa algoritmo genetico para ajustar pesos.

## 8. Por que vale aprender backprop mesmo assim

Porque ele e o metodo mais central no treino moderno de redes neurais.

Mesmo que seu projeto atual nao use, entender a diferenca te da maturidade tecnica.

## 9. Contraste claro

### Backpropagation

- atualiza pesos com base no erro e gradiente
- muito usado em supervised learning e deep learning

### Algoritmo genetico

- avalia conjuntos inteiros de pesos
- seleciona os melhores
- cria novas redes por mutacao e crossover

## 10. O que voce precisa dominar neste modulo

- pesos como parametros
- bias como deslocamento
- forward como computacao da saida
- backprop como uma forma de treino
- por que este projeto escolheu outro caminho

## Exercicios

### Exercicio 1

Escreva uma resposta curta para: "qual a diferenca entre usar uma rede neural e treinar uma rede neural?"

### Exercicio 2

Implemente uma camada neural simples e explique onde estao pesos, bias e forward no codigo.
