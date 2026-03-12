# Modulo NN 08: Como treinar uma rede neural

Este modulo cobre de forma explicita como uma rede neural e treinada em geral, alem do caso evolutivo do projeto.

## 1. O que e treinar

Treinar e ajustar parametros para melhorar desempenho em uma tarefa.

## 2. Treino supervisionado classico

Fluxo tipico:

1. voce tem entradas e respostas desejadas
2. faz forward pass
3. calcula loss
4. calcula gradientes via backpropagation
5. atualiza pesos com um otimizador
6. repete

## 3. Componentes centrais

- dataset
- batch
- epoca
- loss
- gradiente
- otimizador
- learning rate

## 4. Treino evolutivo

No projeto atual, o treino acontece sem backprop:

- varias redes sao testadas
- as melhores sobrevivem
- a nova geracao surge por copia, crossover e mutacao

## 5. Modelos nao treinados

Uma rede nao treinada ainda executa forward, mas normalmente produz comportamento aleatorio ou inutil.

## Exercicios

### Exercicio 1

Descreva o pipeline completo de treino supervisionado em 6 passos.

### Exercicio 2

Descreva o pipeline completo de treino evolutivo deste projeto em 6 passos.
