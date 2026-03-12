# Modulo NN 09: Modelos treinados, nao treinados e transfer learning

Voce pediu explicitamente entender modelo treinado e nao treinado. Este modulo organiza isso.

## 1. Modelo nao treinado

Tem arquitetura e parametros iniciais, mas ainda nao passou por um processo de ajuste util.

## 2. Modelo treinado

Ja passou por um processo de ajuste e reteve parametros que produzem desempenho melhor.

## 3. Checkpoints

Frequentemente voce salva estados intermediarios durante treino.

## 4. Transfer learning

Transfer learning e reaproveitar um modelo treinado em outra tarefa para acelerar uma nova tarefa.

## 5. Fine-tuning

Fine-tuning e continuar treinando um modelo pretreinado para adaptacao.

## 6. No contexto deste projeto

O projeto atual nao faz transfer learning. Cada rede comeca aleatoria ou vem de geracoes anteriores via evolucao.

## Exercicios

### Exercicio 1

Explique a diferenca entre modelo nao treinado, modelo treinado e checkpoint.

### Exercicio 2

Explique por que transfer learning nao e o mecanismo central deste projeto, mas pode ser central em outros problemas.
