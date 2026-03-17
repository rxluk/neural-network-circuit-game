# Modulo Bibliotecas 02: Matplotlib para simulacao e visualizacao

Muita gente conhece Matplotlib so por graficos estaticos. Este projeto mostra um uso mais interessante: visualizacao em tempo real de um sistema aprendendo.

Objetivo deste modulo: voce conseguir abrir uma janela, desenhar a pista, animar um agente e mostrar metricas.

## 1. O papel do Matplotlib aqui

Ele desenha:

- pista
- carros
- sensores
- cards de informacao
- grafico de progresso de aprendizado
- rede neural do melhor agente

Em resumo: ele e a "tela" do seu laboratorio de IA.

## 2. Conceitos fundamentais

- `figure`: janela geral
- `axes`: areas de desenho
- `plot`: linhas
- `scatter`: pontos
- `patches`: formas geometricas
- `collections`: grupos de formas
- eventos de teclado e mouse

Pense assim:

- `figure` = a casa
- `axes` = os comodos
- `artists` (linhas, patches, textos) = moveis

## 3. Objetos concretos usados no projeto

- `Rectangle`
- `Polygon`
- `PathPatch`
- `FancyBboxPatch`
- `Line2D`
- `PolyCollection`
- `LineCollection`

Voce nao precisa decorar todos de uma vez.
Comece com `plot`, `scatter`, `Polygon` e `Rectangle`.

## 4. O que sao patches

Sao objetos geometricos desenhaveis. Muito uteis quando voce quer construir UI ou formas customizadas.

Exemplo: carro como triangulo.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, ax = plt.subplots()
triangulo = Polygon([[0, 0], [1, 0.2], [0, 0.4]], closed=True, color="tab:blue")
ax.add_patch(triangulo)
ax.set_aspect("equal")
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 2)
plt.show()
```

## 5. Evento e interatividade

No projeto, Matplotlib nao e passivo. Ele recebe eventos:

- teclado
- clique do mouse
- timer para atualizar animacao

Isso e crucial para fazer o simulador e o editor de pista.

Exemplo minimo de teclado:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
texto = ax.text(0.5, 0.5, "pressione seta esquerda/direita", ha="center")

def on_key(event):
	if event.key == "left":
		texto.set_text("virando para esquerda")
	elif event.key == "right":
		texto.set_text("virando para direita")
	fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
```

## 6. Animacao por timer

O projeto usa um timer do canvas para chamar a atualizacao continuamente.

Conceitualmente, isso cria o loop visual da simulacao.

Exemplo copy-paste de animacao simples:

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
linha, = ax.plot([], [], lw=2)

x = np.linspace(0, 10, 400)
fase = {"valor": 0.0}

def atualizar():
	fase["valor"] += 0.2
	y = np.sin(x + fase["valor"])
	linha.set_data(x, y)
	fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(atualizar)
timer.start()

plt.show()
```

## 7. Blitting

O projeto usa estrategia de fundo estatico + redesenho do dinamico para renderizar mais eficientemente.

Isso vale estudar com calma, porque e uma tecnica importante quando a cena tem partes fixas e partes animadas.

Resumo pratico:

- desenha fundo uma vez
- atualiza apenas objetos que mudam
- melhora FPS em cenas com muitos elementos

## 8. Quando Matplotlib e uma boa escolha

- prototipos cientificos
- simulacoes didaticas
- dashboards simples
- visualizacao de treinamento

Tambem e excelente para aula, estudo e depuracao visual.

## 9. Quando talvez nao seja a melhor escolha

- jogos pesados em tempo real
- fisica muito complexa em alta taxa de quadros
- interfaces muito interativas e comerciais

Para jogo comercial, engines especificas tendem a ser melhores.

## 10. Licao pratica

Para este tipo de projeto, Matplotlib e suficiente e pedagogicamente excelente.

## 11. Mini dashboard em 2 paineis

Exemplo: cena da simulacao + grafico de score.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, (ax_scene, ax_score) = plt.subplots(1, 2, figsize=(9, 4))

ax_scene.set_title("Cena")
ax_scene.set_xlim(0, 10)
ax_scene.set_ylim(0, 10)
ponto, = ax_scene.plot([1], [1], "o", color="tab:red")

ax_score.set_title("Score")
scores = []
linha_score, = ax_score.plot([], [], color="tab:green")
ax_score.set_xlim(0, 100)
ax_score.set_ylim(0, 100)

estado = {"t": 0}

def atualizar():
	estado["t"] += 1
	x = 1 + 0.05 * estado["t"]
	y = 5 + 2 * np.sin(0.1 * estado["t"])
	ponto.set_data([x], [y])

	score = min(100, estado["t"])
	scores.append(score)
	linha_score.set_data(range(len(scores)), scores)

	fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=40)
timer.add_callback(atualizar)
timer.start()

plt.tight_layout()
plt.show()
```

## 12. Checklist de visualizacao madura

- separou calculo da simulacao e desenho?
- atualiza apenas o necessario por frame?
- organiza estado visual em estruturas claras?
- mostra metricas importantes (geracao, score, vivos)?
- trata eventos de teclado/mouse sem misturar regra de treino?

## Exercicios

### Exercicio 1

Crie uma figura com dois `axes`: um para desenhar um carro como triangulo e outro para desenhar um grafico de score.

### Exercicio 2

Implemente uma pequena animacao em Matplotlib com um ponto se movendo por 100 frames, atualizando o grafico de velocidade ao lado.

### Exercicio 3

Desenhe a pista como duas linhas (borda interna e externa) e plote os sensores de um carro como segmentos.
