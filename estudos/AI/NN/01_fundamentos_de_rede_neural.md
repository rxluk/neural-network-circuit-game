# Modulo NN 01: Fundamentos de rede neural

## 1. A definicao simples e correta

Uma rede neural artificial e um conjunto de transformacoes numericas parametrizadas que mapeia entradas em saidas.

Essa definicao e seca, mas muito precisa.

## 2. O que entra e o que sai no projeto

Entradas:

- 7 sensores de distancia
- 1 velocidade

Saidas:

- comando de virada
- comando de aceleracao

## 3. Neuronio artificial

Cada neuronio faz uma soma ponderada e uma ativacao:

$$
z = x_1w_1 + x_2w_2 + ... + x_nw_n + b
$$

$$
a = f(z)
$$

## 4. Camadas

Uma rede simples pode ser vista assim:

```mermaid
flowchart LR
    A[Entradas] --> B[Camada oculta]
    B --> C[Saida]
```

No projeto:

$$
8 \rightarrow 14 \rightarrow 2
$$

## 5. O que a rede aprende

Ela aprende combinacoes de pesos e bias que produzem boas acoes para os estados observados.

## 6. O que a rede nao sabe sozinha

Ela nao sabe:

- o que e pista
- o que e bom ou ruim
- o que e reward

Ela depende do sistema em volta.

## 7. A intuicao mais importante

Rede neural e uma maquina de transformar sinais. O comportamento vem da composicao dessas transformacoes.

## Exercicios

### Exercicio 1

Explique por que uma rede neural sem ambiente, entrada e criterio de sucesso nao faz nada util sozinha.

### Exercicio 2

Desenhe em papel uma rede `4 -> 3 -> 2` e rotule entradas, camada oculta e saidas.
