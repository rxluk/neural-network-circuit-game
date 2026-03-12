# Modulo NN 10: Salvando, carregando e versionando modelos

Treinar sem salvar direito e desperdiçar parte do trabalho.

## 1. O que salvar

- pesos
- bias
- arquitetura
- hiperparametros
- metricas principais
- contexto do experimento

## 2. O que o projeto atual salva

Ele salva pesos e resultados em JSON, o que faz sentido para uma rede pequena e um projeto didatico.

## 3. Salvando em projetos NumPy puros

Opcoes naturais:

- JSON
- `.npy`
- `.npz`

## 4. Em frameworks modernos

Em bibliotecas como PyTorch, voce costuma salvar `state_dict` e metadados.

## 5. Versionamento de modelo

Boa pratica:

- nomear versoes
- registrar hiperparametros
- registrar metrica atingida
- salvar data e observacoes

## Exercicios

### Exercicio 1

Defina um formato minimo de arquivo para salvar uma mini-rede sua com pesos, bias, nome da arquitetura e data.

### Exercicio 2

Explique o que precisaria ser salvo para voce conseguir retomar um experimento de treino semanas depois sem se perder.
