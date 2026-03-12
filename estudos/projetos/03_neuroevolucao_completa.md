# Projeto 03: Neuroevolucao completa

## Objetivo

Construir uma versao propria bem proxima do projeto estudado.

## O que voce vai aprender

- populacao de agentes
- fitness por episodio
- elitismo
- crossover
- mutacao
- metricas por geracao
- visualizacao do progresso

## Escopo sugerido

- pista fechada
- 7 sensores + velocidade
- rede pequena MLP
- algoritmo genetico completo
- visualizacao com Matplotlib

## Estrutura sugerida

```text
main.py
config.json
pista.json
sim/
  track.py
  neural_network.py
  evolution.py
  visualizacao.py
```

## Critero de pronto

- populacao roda por geracao
- melhores agentes sao selecionados
- nova geracao e criada por copia, crossover e mutacao
- metricas mostram melhoria ao longo do tempo
- o melhor agente passa a fazer curvas e progredir melhor

## Extensoes

- editor de pista
- salvar melhor cerebro
- mutacao adaptativa
- painel lateral com rede neural e metricas
