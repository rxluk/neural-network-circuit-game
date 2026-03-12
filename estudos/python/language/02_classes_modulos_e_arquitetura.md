# Modulo Python 02: Classes, modulos e arquitetura de projeto

Este modulo trata da habilidade que mais separa um prototipo baguncado de um sistema estudavel: arquitetura.

## 1. O que e um modulo em Python

Um modulo e um arquivo `.py`.

Uma pasta com varios modulos relacionados forma um pacote.

No projeto, a pasta `sim/` funciona como o pacote principal do dominio.

## 2. Por que modularizar

Porque um sistema como este tem responsabilidades diferentes:

- geometria
- fisica
- modelo neural
- evolucao
- visualizacao

Se voce mistura tudo, perde a capacidade de raciocinar localmente.

## 3. Quando criar uma classe

Crie classe quando fizer sentido modelar uma entidade com estado e comportamento relacionado.

Faz sentido para:

- carro
- rede neural
- simulador

Nao faz tanto sentido para tudo. Funcoes utilitarias tambem sao importantes.

## 4. Estado vs comportamento

Exemplo do carro:

- estado: `x`, `y`, `angulo`, `velocidade`, `vivo`
- comportamento: `mover()`, `get_sensores()`, `checar_colisao()`

## 5. Boa divisao para o seu proprio projeto

Uma arquitetura simples e boa:

```text
seu_projeto/
  main.py
  config.json
  sim/
    track.py
    agent.py
    network.py
    evolution.py
    render.py
```

## 6. Importacoes entre modulos

Python deixa facil ligar modulos:

```python
from sim.track import CarrinhoIA
from sim.network import RedeNeural
```

Mas cuidado para nao criar dependencia circular desnecessaria.

## 7. Coesao e acoplamento

### Coesao boa

Quando um modulo tem foco claro.

### Acoplamento ruim

Quando um modulo precisa conhecer detalhes demais de outro.

O projeto faz algo bom aqui: a visualizacao usa o simulador, mas o nucleo nao depende da UI para existir.

## 8. Separar motor e interface

Essa e uma licao muito boa do projeto atual.

Se o motor de simulacao e independente da visualizacao, voce pode:

- testar mais facil
- rodar headless
- trocar a interface depois

## 9. Heranca usada com criterio

No projeto, `SimuladorAprendizado` estende `SimuladorBase`.

Isso faz sentido porque:

- o comportamento base de simulacao e reutilizado
- a camada visual adiciona UI e renderizacao

## 10. Sinais de arquitetura ruim

- arquivo gigante faz tudo
- a rede conhece detalhes da interface
- o carro salva arquivo JSON sozinho
- a visualizacao decide regra de fitness

## 11. Arquitetura orientada a evolucao do projeto

Um projeto bom nao nasce perfeito. Ele cresce.

Boa estrategia:

1. uma versao simples em 1 ou 2 arquivos
2. separar quando a responsabilidade ficar clara
3. mover configuracao para JSON
4. so depois adicionar refinamentos de performance ou UI

## 12. Como ler classes tecnicas sem se perder

Para cada classe, responda:

- qual entidade ela representa?
- quais atributos guardam estado?
- quais metodos alteram esse estado?
- de quem ela depende?

## Exercicios

### Exercicio 1

Desenhe a arquitetura ideal para uma versao reduzida deste projeto com 6 arquivos.

### Exercicio 2

Refatore um mini-script seu para separar `agente`, `ambiente` e `treino` em modulos diferentes.
