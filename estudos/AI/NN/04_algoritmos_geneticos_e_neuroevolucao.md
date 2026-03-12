# Modulo NN 04: Algoritmos geneticos e neuroevolucao

Este modulo fecha o mecanismo de aprendizagem usado aqui.

## 1. O que e algoritmo genetico

E uma estrategia populacional de busca.

Voce mantem varias solucoes, mede desempenho, seleciona melhores, mistura e muta.

## 2. O que e neuroevolucao

Neuroevolucao e usar algoritmo evolutivo para otimizar redes neurais.

No projeto, o que evolui sao os pesos e bias da rede.

## 3. Elementos centrais

- populacao
- fitness
- selecao
- elitismo
- crossover
- mutacao

## 4. Fitness neste projeto

Fitness vem de pontos por:

- progresso
- velocidade
- voltas

e penalidades por:

- colisao
- contramao
- lentidao
- falta de progresso

## 5. Crossover

Mistura genes de pais diferentes.

## 6. Mutacao

Adiciona perturbacao aleatoria aos parametros.

## 7. Por que funciona

Porque algumas combinacoes de pesos produzem comportamento melhor. Essas combinacoes passam a ter mais descendencia.

## 8. Limites

- pode demorar
- pode ser instavel
- depende muito de reward bem desenhada

## 9. Diferenca para backprop

Backprop ajusta peso por gradiente.

Neuroevolucao busca boas configuracoes de pesos pela selecao de comportamento.

## 10. Quando usar

Faz muito sentido em simulacoes pequenas, problemas nao diferenciaveis e estudos didaticos como este.

## Exercicios

### Exercicio 1

Explique em suas palavras por que uma populacao inteira pode aprender algo que um individuo isolado aleatorio nao aprende rapidamente.

### Exercicio 2

Implemente um mini-algoritmo genetico para maximizar uma funcao simples e depois adapte a ideia para 2 ou 3 pesos de uma mini-rede.
