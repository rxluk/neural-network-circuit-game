# Modulo Bibliotecas 04: Serializacao de modelos e formatos de arquivo

Uma hora voce vai treinar algo e vai querer salvar. Este modulo responde: salvar em que arquivo e de que jeito?

## 1. O que significa serializar

Serializar e transformar um estado em algo persistivel em disco para ser restaurado depois.

## 2. O que voce pode querer salvar

- pesos e bias
- hiperparametros
- metricas de treino
- arquitetura do modelo
- seed e contexto do experimento

## 3. Formatos comuns

### JSON

Bom para:

- pesos pequenos
- hiperparametros
- resultados legiveis

### NumPy `.npy` e `.npz`

Muito bons para arrays e matrizes.

### Pickle

Conveniente, mas menos transparente e mais sensivel a contexto de ambiente e seguranca.

## 4. O que o projeto atual faz

Ele salva resultados e pesos em JSON. Para este projeto, isso e aceitavel porque a rede e pequena e o foco e estudo.

## 5. O que vale salvar junto do modelo

- nome ou versao da arquitetura
- shape esperado das entradas
- configuracao de treino
- data
- melhor metrica atingida

## Exercicios

### Exercicio 1

Salve uma mini-rede em JSON e recarregue os pesos depois.

### Exercicio 2

Salve a mesma mini-rede em `.npz` e compare a experiencia com JSON.
