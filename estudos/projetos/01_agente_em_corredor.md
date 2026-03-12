# Projeto 01: Agente em corredor

## Objetivo

Construir um agente minimo que anda em um corredor simples sem bater.

## O que voce vai aprender

- organizacao basica do projeto
- movimento 2D simples
- sensores minimos
- colisao simples
- reward simples

## Escopo sugerido

- pista em forma de corredor reto
- 3 sensores: esquerda, frente, direita
- sem rede neural no inicio, podendo usar regras manuais
- depois trocar o controlador por uma rede pequena

## Estrutura sugerida

```text
main.py
config.json
sim/
  track.py
  agent.py
  network.py
```

## Critero de pronto

- o agente se move
- detecta parede
- os sensores retornam numeros coerentes
- ha uma reward minima funcionando

## Extensoes

- desenhar com Matplotlib
- logar score por episodio
