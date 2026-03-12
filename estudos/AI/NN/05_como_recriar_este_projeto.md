# Modulo NN 05: Como recriar este projeto sozinho

Este modulo e a ponte final entre estudo e construcao.

## 1. Ordem correta de implementacao

1. faca um ambiente simples
2. implemente movimento do carro
3. implemente sensores
4. implemente uma rede pequena
5. implemente reward e penalidade
6. implemente algoritmo genetico
7. adicione visualizacao
8. refine pista, metricas e performance

## 2. Estrutura minima recomendada

```text
meu_projeto/
  main.py
  config.json
  pista.json
  sim/
    track.py
    neural_network.py
    evolution.py
    visualizacao.py
```

## 3. Primeira versao que eu recomendo

- 3 sensores
- pista simples
- rede `4 -> 6 -> 2`
- populacao pequena
- reward simples por sobreviver e progredir

## 4. Segunda versao

- 7 sensores
- pista fechada
- reward com progresso e velocidade
- graficos de geracao

## 5. Terceira versao

- editor de pista
- mutacao adaptativa
- salvar melhor cerebro
- painel de visualizacao

## 6. Checklist de dominio real

Voce estara pronto quando conseguir fazer sem cola:

- medir sensores
- converter saida da rede em acao
- detectar colisao
- calcular fitness
- gerar nova populacao
- acompanhar evolucao entre geracoes

## 7. Erros que voce deve evitar

- comecar com pista complexa demais
- criar reward enorme antes da versao simples funcionar
- fazer visual bonito antes de o motor estar correto
- aumentar a rede sem saber por que

## 8. O que eu faria se estivesse aprendendo hoje

Eu implementaria em 4 mini-marcos:

- marco 1: carro anda
- marco 2: carro sente
- marco 3: carro decide
- marco 4: populacao evolui

## Exercicios

### Exercicio 1

Escreva um plano seu de implementacao em 7 passos para recriar esse projeto do zero em escala menor.

### Exercicio 2

Implemente a versao minima: um agente com 3 sensores, uma rede pequena e um algoritmo genetico para evitar bater em uma pista simples.
