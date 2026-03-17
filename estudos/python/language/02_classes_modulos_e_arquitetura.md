# Modulo Python 02: Classes, modulos e arquitetura de projeto

Este modulo trata da habilidade que mais separa um prototipo baguncado de um sistema estudavel: arquitetura.

## 1. O que e um modulo em Python

Um modulo e um arquivo `.py`.

Uma pasta com varios modulos relacionados forma um pacote.

No projeto, a pasta `sim/` funciona como o pacote principal do dominio.

Pense em modulo como "uma caixa com um assunto".

- `sim/track.py`: regra de pista e geometria
- `sim/neural_network.py`: rede neural
- `sim/simulacao.py`: ciclo de simulacao/treino
- `sim/visualizacao.py`: interface e desenho

## 2. Por que modularizar

Porque um sistema como este tem responsabilidades diferentes:

- geometria
- fisica
- modelo neural
- evolucao
- visualizacao

Se voce mistura tudo, perde a capacidade de raciocinar localmente.

Para uma crianca entender: imagine uma escola.

- sala de aula nao e cozinha
- cozinha nao e secretaria
- cada lugar tem seu papel

Codigo funciona igual: cada arquivo precisa de um papel claro.

## 3. Quando criar uma classe

Crie classe quando fizer sentido modelar uma entidade com estado e comportamento relacionado.

Faz sentido para:

- carro
- rede neural
- simulador

Nao faz tanto sentido para tudo. Funcoes utilitarias tambem sao importantes.

Regra simples:

- se precisa lembrar estado ao longo do tempo, classe
- se so transforma entrada em saida, funcao

## 4. Estado vs comportamento

Exemplo do carro:

- estado: `x`, `y`, `angulo`, `velocidade`, `vivo`
- comportamento: `mover()`, `get_sensores()`, `checar_colisao()`

Estado = "o que o objeto sabe".
Comportamento = "o que o objeto faz".

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

### Versao minima para comecar

```text
seu_projeto/
  main.py
  sim/
    agente.py
    ambiente.py
```

Quando crescer, voce quebra em mais arquivos.

## 6. Importacoes entre modulos

Python deixa facil ligar modulos:

```python
from sim.track import CarrinhoIA
from sim.network import RedeNeural
```

Mas cuidado para nao criar dependencia circular desnecessaria.

Dependencia circular e quando A importa B e B importa A.
Sintoma comum: erro de import no startup.

## 7. Coesao e acoplamento

### Coesao boa

Quando um modulo tem foco claro.

### Acoplamento ruim

Quando um modulo precisa conhecer detalhes demais de outro.

O projeto faz algo bom aqui: a visualizacao usa o simulador, mas o nucleo nao depende da UI para existir.

Isso permite rodar treino sem abrir janela.

## 8. Separar motor e interface

Essa e uma licao muito boa do projeto atual.

Se o motor de simulacao e independente da visualizacao, voce pode:

- testar mais facil
- rodar headless
- trocar a interface depois

Exemplo real: hoje Matplotlib, amanha Pygame ou frontend web.
O motor continua valendo.

## 9. Heranca usada com criterio

No projeto, `SimuladorAprendizado` estende `SimuladorBase`.

Isso faz sentido porque:

- o comportamento base de simulacao e reutilizado
- a camada visual adiciona UI e renderizacao

Se a heranca virar confusa, prefira composicao.
Composicao = um objeto possui outro objeto.

## 10. Sinais de arquitetura ruim

- arquivo gigante faz tudo
- a rede conhece detalhes da interface
- o carro salva arquivo JSON sozinho
- a visualizacao decide regra de fitness

Outros sinais:

- metodos com 200+ linhas
- imports demais no mesmo arquivo
- variaveis globais espalhadas

## 11. Arquitetura orientada a evolucao do projeto

Um projeto bom nao nasce perfeito. Ele cresce.

Boa estrategia:

1. uma versao simples em 1 ou 2 arquivos
2. separar quando a responsabilidade ficar clara
3. mover configuracao para JSON
4. so depois adicionar refinamentos de performance ou UI

Essa ordem evita overengineering.

## 12. Como ler classes tecnicas sem se perder

Para cada classe, responda:

- qual entidade ela representa?
- quais atributos guardam estado?
- quais metodos alteram esse estado?
- de quem ela depende?

## 13. Exemplo concreto: script baguncado -> arquitetura limpa

### Antes (tudo em um arquivo)

```python
# main.py (ruim para crescer)
import numpy as np

pesos = np.random.randn(8, 2)
carro_x, carro_y = 0.0, 0.0

def decidir(sensores):
  return sensores @ pesos

def atualizar_posicao(acao):
  global carro_x, carro_y
  carro_x += acao[0]
  carro_y += acao[1]
```

### Depois (separado por responsabilidade)

```python
# sim/network.py
import numpy as np

class RedeNeural:
  def __init__(self, n_in=8, n_out=2):
    self.W = np.random.randn(n_in, n_out)

  def forward(self, sensores):
    return sensores @ self.W
```

```python
# sim/agent.py
class Agente:
  def __init__(self, rede):
    self.rede = rede
    self.x = 0.0
    self.y = 0.0

  def passo(self, sensores):
    acao = self.rede.forward(sensores)
    self.x += float(acao[0])
    self.y += float(acao[1])
```

```python
# main.py
import numpy as np
from sim.agent import Agente
from sim.network import RedeNeural

agente = Agente(RedeNeural())
agente.passo(np.ones(8))
```

Perceba o ganho:

- cada arquivo faz uma coisa
- testar fica mais facil
- trocar rede ou agente nao quebra tudo

## 14. Mapa mental para quem vem de Java/PHP

- classe Java `Car`: classe Python `Carro`
- pacote Java `com.projeto.sim`: pasta Python `sim/`
- arquivo PHP com funcoes utilitarias: modulo Python utilitario

O principal ajuste mental e: em Python, modulo tem muito protagonismo.

## 15. Mini checklist de arquitetura para seu proximo projeto

- existe separacao entre ambiente, agente e treino?
- rede neural esta desacoplada da UI?
- configuracoes estao fora do codigo (JSON/YAML)?
- consigo testar o motor sem abrir interface?
- cada arquivo responde a uma pergunta clara?

## Exercicios

### Exercicio 1

Desenhe a arquitetura ideal para uma versao reduzida deste projeto com 6 arquivos.

### Exercicio 2

Refatore um mini-script seu para separar `agente`, `ambiente` e `treino` em modulos diferentes.

### Exercicio 3

Pegue uma classe do projeto e descreva:

- estado
- comportamento
- dependencias
- por que ela deveria existir como classe (e nao so funcao)
