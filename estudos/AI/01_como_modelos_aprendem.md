# Modulo IA 01: Como modelos aprendem

Quando alguem fala que um modelo "aprende", a pergunta certa e: como os parametros dele melhoram?

## 1. Aprender nao e magia

Aprender, em ML, significa ajustar parametros ou estruturas internas de modo que o desempenho melhore em alguma tarefa.

## 2. Formas comuns de aprendizagem

### Supervisionada

Tem pares entrada-saida desejada.

### Nao supervisionada

Busca estrutura nos dados sem rotulos explicitos.

### Por reforco

Um agente interage com ambiente e recebe recompensas.

### Evolutiva

Varios candidatos sao avaliados e os melhores geram novos candidatos.

## 3. O que sao parametros nesse contexto

Em redes neurais, tipicamente:

- pesos
- bias

Aprender e encontrar valores melhores para esses parametros.

## 4. Como isso acontece no projeto

O projeto avalia o comportamento do carro inteiro. Depois seleciona redes melhores e gera novas redes via copia, crossover e mutacao.

## 5. O que torna uma forma de aprendizagem apropriada

- tipo de dado
- tipo de ambiente
- disponibilidade de rotulos
- facilidade ou nao de calcular gradientes
- custo computacional

## 6. Licao principal

O verbo "aprender" so faz sentido quando voce sabe qual mecanismo esta alterando o modelo.

## Exercicios

### Exercicio 1

Diga com suas palavras a diferenca entre aprender por gradiente e aprender por algoritmo genetico.

### Exercicio 2

Escolha um problema qualquer e diga qual forma de aprendizagem faria mais sentido para ele e por que.
