# Modulo Bibliotecas 02: Matplotlib para simulacao e visualizacao

Muita gente conhece Matplotlib so por graficos estaticos. Este projeto mostra um uso mais interessante: visualizacao em tempo real de um sistema aprendendo.

## 1. O papel do Matplotlib aqui

Ele desenha:

- pista
- carros
- sensores
- cards de informacao
- grafico de progresso de aprendizado
- rede neural do melhor agente

## 2. Conceitos fundamentais

- `figure`: janela geral
- `axes`: areas de desenho
- `plot`: linhas
- `scatter`: pontos
- `patches`: formas geometricas
- `collections`: grupos de formas
- eventos de teclado e mouse

## 3. Objetos concretos usados no projeto

- `Rectangle`
- `Polygon`
- `PathPatch`
- `FancyBboxPatch`
- `Line2D`
- `PolyCollection`
- `LineCollection`

## 4. O que sao patches

Sao objetos geometricos desenhaveis. Muito uteis quando voce quer construir UI ou formas customizadas.

## 5. Evento e interatividade

No projeto, Matplotlib nao e passivo. Ele recebe eventos:

- teclado
- clique do mouse
- timer para atualizar animacao

Isso e crucial para fazer o simulador e o editor de pista.

## 6. Animacao por timer

O projeto usa um timer do canvas para chamar a atualizacao continuamente.

Conceitualmente, isso cria o loop visual da simulacao.

## 7. Blitting

O projeto usa estrategia de fundo estatico + redesenho do dinamico para renderizar mais eficientemente.

Isso vale estudar com calma, porque e uma tecnica importante quando a cena tem partes fixas e partes animadas.

## 8. Quando Matplotlib e uma boa escolha

- prototipos cientificos
- simulacoes didaticas
- dashboards simples
- visualizacao de treinamento

## 9. Quando talvez nao seja a melhor escolha

- jogos pesados em tempo real
- fisica muito complexa em alta taxa de quadros
- interfaces muito interativas e comerciais

## 10. Licao pratica

Para este tipo de projeto, Matplotlib e suficiente e pedagogicamente excelente.

## Exercicios

### Exercicio 1

Crie uma figura com dois `axes`: um para desenhar um carro como triangulo e outro para desenhar um grafico de score.

### Exercicio 2

Implemente uma pequena animacao em Matplotlib com um ponto se movendo por 100 frames, atualizando o grafico de velocidade ao lado.
