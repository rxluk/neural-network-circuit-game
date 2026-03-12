# Guia de Estudo do Projeto

Este projeto e uma simulacao de um carro que aprende a dirigir em uma pista usando:

- uma rede neural pequena feita manualmente
- um algoritmo genetico para ajustar os pesos da rede
- fisica simples do carro
- sensores de distancia
- visualizacao em tempo real com Matplotlib

Nao ha backpropagation, nao ha treino com gradiente, nao ha dataset rotulado. O aprendizado acontece por selecao evolutiva: redes melhores sobrevivem e geram novas redes com mutacoes.

## 1. O que o projeto faz

Em cada geracao:

1. varios carros nascem com redes neurais diferentes
2. cada carro observa a pista com sensores
3. a rede neural decide quanto virar e quanto acelerar
4. os carros recebem pontos por progresso, voltas e velocidade
5. carros ruins morrem ou ficam com fitness baixo
6. os melhores viram pais da proxima geracao
7. novos filhos sao criados com copia, crossover e mutacao dos pesos

Depois de varias geracoes, algumas redes passam a controlar o carro melhor.

## 2. Isso e IA? Isso e machine learning?

Sim, e os dois.

- IA: no sentido amplo, porque existe um agente tomando decisoes autonomas no ambiente
- Machine Learning: porque o comportamento melhora a partir de experiencia, sem regras fixas programadas curva por curva

Mais especificamente, isso entra em:

- neuroevolucao: redes neurais treinadas por algoritmo evolutivo
- computacao evolutiva: selecao, crossover e mutacao
- aprendizagem por interacao com ambiente: parecido com reinforcement learning no espirito, mas sem usar Q-learning, policy gradient ou backpropagation

Entao o nome tecnico mais preciso aqui seria algo como:

- simulacao de neuroevolucao para controle de agente

## 3. Arquitetura do projeto

### Entrada principal

O arquivo [rede_neural_jogo.py](/home/luk/neural-network-circuit-game/rede_neural_jogo.py) apenas cria o simulador visual e inicia a animacao.

### Nucleo da simulacao

Os arquivos principais do nucleo sao:

- [sim/neural_network.py](/home/luk/neural-network-circuit-game/sim/neural_network.py): rede neural e operadores geneticos
- [sim/simulacao.py](/home/luk/neural-network-circuit-game/sim/simulacao.py): loop evolutivo
- [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py): pista, sensores, colisao, fisica e entidade do carro

### Visualizacao

O arquivo [sim/visualizacao.py](/home/luk/neural-network-circuit-game/sim/visualizacao.py) desenha tudo com Matplotlib em tempo real.

### Editor da pista

O arquivo [editor_pista.py](/home/luk/neural-network-circuit-game/editor_pista.py) permite criar uma pista interativamente e salvar em [pista.json](/home/luk/neural-network-circuit-game/pista.json).

### Configuracao

O arquivo [config.json](/home/luk/neural-network-circuit-game/config.json) concentra quase todos os hiperparametros.

## 4. Como a pista e representada

O projeto nao usa imagem de pista. A pista e descrita geometricamente.

### Pontos de controle

Em [pista.json](/home/luk/neural-network-circuit-game/pista.json), a pista e definida por:

- nome
- pontos_controle
- largura_pista
- configuracao de largada/chegada

### Suavizacao da pista

Em [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py) e [editor_pista.py](/home/luk/neural-network-circuit-game/editor_pista.py), os pontos sao suavizados com spline Catmull-Rom.

Isso cria uma centerline continua, isto e, uma linha central curva da pista. Depois o codigo gera duas bordas offsetadas a partir dessa linha central.

### SDF: Signed Distance Field

Um detalhe muito bom do projeto: a verificacao se o ponto esta dentro da pista usa uma grade pre-computada de distancias, chamada SDF.

Ideia:

- para muitos pontos no plano, calcula-se a distancia ate a centerline
- se a distancia for menor ou igual a metade da largura da pista, o ponto esta dentro

Isso torna a consulta de colisao muito rapida.

## 5. O que e o carro no projeto

Cada carro e uma instancia de `CarrinhoIA`, definida em [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py).

Ele guarda estado como:

- posicao `x`, `y`
- angulo
- velocidade
- se esta vivo
- tempo de vida
- pontos acumulados
- quantidade de voltas
- melhor tempo de volta
- trilha percorrida
- eventos e motivo da morte

Esse carro nao aprende sozinho por formula magica. Quem decide as acoes e a rede neural associada a ele.

## 6. Como funcionam os sensores

O carro possui 7 sensores de distancia em leque:

- -90
- -45
- -22.5
- 0
- +22.5
- +45
- +90

Eles estao definidos em `CarrinhoIA._SENSOR_OFFSETS` em [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py).

### Como o sensor mede distancia

Para cada angulo:

1. o codigo cria varios pontos ao longo de um raio saindo do carro
2. testa quais pontos ainda estao dentro da pista
3. quando encontra o primeiro ponto fora, toma aquela distancia como leitura do sensor

As distancias sao normalizadas dividindo por 15.0.

### Entrada total da rede

O metodo `get_sensores()` retorna 8 valores:

- 7 distancias normalizadas
- 1 valor de velocidade normalizada

Por isso a rede tem 8 entradas.

## 7. Ele fez uma rede neural mesmo?

Sim. E uma rede neural feedforward totalmente conectada, implementada manualmente com NumPy em [sim/neural_network.py](/home/luk/neural-network-circuit-game/sim/neural_network.py).

Arquitetura:

- 8 entradas
- 14 neuronios na camada oculta
- 2 saidas

Em forma resumida:

$$
8 \rightarrow 14 \rightarrow 2
$$

## 8. O que sao pesos e vieses

Aqui esta a base da rede neural.

### Pesos

Pesos dizem o quanto uma entrada influencia um neuronio.

Exemplo mental:

- se o sensor frontal estiver baixo, talvez isso devesse reduzir aceleracao
- se o sensor da esquerda estiver baixo, talvez isso devesse aumentar virar para a direita

Os pesos sao justamente os numeros que definem essas influencias.

Matematicamente, um neuronio calcula algo como:

$$
z = x_1w_1 + x_2w_2 + \dots + x_nw_n + b
$$

onde:

- $x_i$ sao as entradas
- $w_i$ sao os pesos
- $b$ e o vies

### Vieses

O vies desloca a ativacao do neuronio. Ele funciona como um ajuste independente das entradas.

Sem vies, o neuronio dependeria apenas da soma ponderada das entradas. Com vies, ele pode ser mais ou menos facil de ativar.

## 9. Como os pesos foram criados no codigo

No construtor de `RedeNeuralCarrinho`, em [sim/neural_network.py](/home/luk/neural-network-circuit-game/sim/neural_network.py), temos:

- `W1` com shape `(8, 14)`
- `b1` com shape `(14,)`
- `W2` com shape `(14, 2)`
- `b2` com shape `(2,)`

Isso significa:

- `W1`: liga as 8 entradas aos 14 neuronios ocultos
- `b1`: vieses da camada oculta
- `W2`: liga os 14 neuronios ocultos as 2 saidas
- `b2`: vieses da camada de saida

Os pesos sao inicializados assim:

$$
W \sim \mathcal{N}(0, 1) \times 0.5
$$

Em palavras:

- pega numeros aleatorios de distribuicao normal
- multiplica por `0.5`
- isso gera pesos pequenos aleatorios no inicio

Os vieses comecam zerados.

## 10. Como a rede toma decisao

O metodo `decidir_acao()` faz o forward pass:

1. pega o vetor de sensores
2. calcula a camada oculta
3. aplica sigmoid
4. calcula a camada de saida
5. aplica sigmoid de novo
6. transforma as saidas em comandos do carro

Formula da primeira camada:

$$
a_1 = \sigma(xW_1 + b_1)
$$

Formula da saida:

$$
a_2 = \sigma(a_1W_2 + b_2)
$$

### Funcao de ativacao

A funcao usada e a sigmoid:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Ela transforma valores em algo entre 0 e 1.

### Saidas da rede

As duas saidas sao interpretadas assim:

- saida 1: direcao
- saida 2: aceleracao/freio

No codigo:

- direcao: `(a2[0] - 0.5) * 2.0`, ficando entre `-1` e `+1`
- aceleracao: `a2[1]`, ficando entre `0` e `1`

Interpretacao:

- `-1`: virar forte para um lado
- `+1`: virar forte para o outro
- `0`: nao virar
- `0` em aceleracao: frear totalmente
- `1` em aceleracao: acelerar totalmente

## 11. Como o carro se move

O metodo `_aplicar_fisica()` em [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py) usa a acao da rede para atualizar o estado do carro.

### Virada

O angulo do carro e atualizado por:

$$
\text{angulo} = \text{angulo} + \text{acao\_virar} \times \text{angulo\_virada\_max}
$$

### Velocidade

Se a saida de aceleracao for pelo menos `0.5`, a velocidade sobe um passo. Caso contrario, a velocidade desce um passo.

Ou seja, a segunda saida funciona como um controle simples de gas/freio, nao como aceleracao fisica continua sofisticada.

### Posicao

Depois, a posicao e atualizada com trigonometria:

$$
x = x + v\cos(\theta)
$$

$$
y = y + v\sin(\theta)
$$

## 12. Onde esta o aprendizado de verdade

O aprendizado nao esta no forward da rede. O forward so usa os pesos atuais.

O aprendizado acontece entre geracoes, em [sim/simulacao.py](/home/luk/neural-network-circuit-game/sim/simulacao.py), dentro de `evoluir_geracao()`.

Esse arquivo faz o papel de algoritmo genetico.

## 13. Como o algoritmo genetico funciona aqui

### Passo 1: cada carro recebe um fitness

O projeto usa `pontos_acumulados` como base do fitness, com ajuste por tempo de volta.

O carro ganha ou perde pontos por varios fatores.

### Recompensas

Entre as recompensas:

- andar perto do centro da pista
- avancar pela centerline
- completar volta
- fazer volta rapida
- manter velocidade boa depois da primeira volta

### Penalidades

Entre as penalidades:

- bater ou sair da pista
- andar na contramao
- ficar muito perto da parede
- ficar devagar demais depois da primeira volta
- passar muitos frames sem progresso
- acumular perda continua de pontos

### Passo 2: seleciona os melhores

Em `evoluir_geracao()`, o codigo escolhe os `top_sobreviventes` com melhor fitness.

Isso e selecao natural artificial.

### Passo 3: elitismo

Os melhores sao copiados diretamente para a nova geracao. Isso preserva boas solucoes.

No codigo, isso acontece com `copiar_de()`.

### Passo 4: novos aleatorios

Algumas redes completamente novas sao adicionadas a cada geracao.

Isso evita que a populacao fique sem diversidade cedo demais.

### Passo 5: crossover

Um filho pode combinar pesos de dois pais.

No metodo `crossover()`, para cada posicao do array de pesos ou vieses, uma mascara aleatoria escolhe se o valor vem do pai A ou do pai B.

Isso se chama crossover uniforme.

### Passo 6: mutacao

Depois do crossover, a rede sofre mutacao com ruido gaussiano:

$$
W = W + \epsilon
$$

onde:

$$
\epsilon \sim \mathcal{N}(0, \text{taxa})
$$

No codigo, isso e feito por `mutar()`.

Em termos simples:

- cada peso e levemente perturbado
- isso cria novos comportamentos
- alguns ficam piores
- alguns melhoram

E assim a populacao vai explorando solucoes.

## 14. Por que isso funciona sem backpropagation?

Porque o projeto nao esta ensinando a rede por exemplo supervisionado. Em vez disso, ele mede o resultado final do comportamento.

Se uma rede leva o carro a um comportamento melhor:

- ela recebe fitness maior
- tem mais chance de influenciar a proxima geracao

Isso e uma busca evolutiva no espaco de pesos.

Em vez de perguntar:

- qual gradiente corrige este peso?

o projeto pergunta:

- quais conjuntos de pesos geram melhor comportamento geral?

## 15. Como o projeto evita carros ruins vivos para sempre

O codigo tem varios mecanismos bons para matar agentes improdutivos:

- colisao imediata ao sair da pista
- morte por falta de progresso na centerline
- morte por perda continua de pontos
- congelamento ou penalizacao por contramao

Isso reduz desperdicio computacional e acelera a evolucao.

## 16. Como a simulacao inteira roda por frame

O fluxo principal por frame e:

1. para cada carro vivo, ler sensores
2. aplicar regras de penalidade preliminares
3. juntar todos os sensores dos sobreviventes
4. fazer um forward batch da rede neural
5. mover os carros com as acoes geradas
6. atualizar progresso, voltas e colisao
7. quando todos morrem, evoluir a geracao

Esse forward em lote esta em `forward_batch()` em [sim/neural_network.py](/home/luk/neural-network-circuit-game/sim/neural_network.py).

Esse ponto e importante: o autor nao fez apenas funcionar, ele tambem otimizou para processar varios carros ao mesmo tempo com NumPy.

## 17. O que foi usado no projeto

### Bibliotecas realmente usadas

- Python
- NumPy
- Matplotlib
- tkinter, apenas para clipboard no editor de pista

### Biblioteca listada mas nao usada em runtime

- PyTorch

O proprio README e o arquivo de build mostram isso. O executavel exclui `torch`, `torchvision` e `torchaudio`.

Entao, para esta versao do projeto, o coracao do sistema foi feito essencialmente com NumPy e Matplotlib.

## 18. O que o autor aprendeu construindo isso

O projeto obriga a entender na pratica:

- como representar uma pista matematicamente
- como sensores viram numeros
- como entradas passam por pesos e ativacoes
- como uma rede gera uma acao
- como definir uma funcao de fitness
- como mutacao e crossover mudam comportamento
- como diversidade e selecao afetam o aprendizado
- como visualizacao ajuda a entender aprendizado emergente

Isso e uma forma excelente de estudar, porque conecta matematica, codigo e comportamento visivel.

## 19. O que voce precisa estudar para chegar nesse nivel

Se eu fosse organizar sua trilha, seria assim.

### Etapa 1: base de Python

Estude:

- funcoes
- classes e objetos
- listas, dicionarios e tuplas
- modulos e organizacao de projeto
- leitura e escrita de JSON

### Etapa 2: algebra linear basica

Voce nao precisa virar matematico primeiro, mas precisa dominar:

- vetores
- matrizes
- produto escalar
- multiplicacao matriz-vetor
- shapes de arrays

Sem isso, rede neural vira decoracao.

### Etapa 3: NumPy

Estude muito:

- arrays
- broadcasting
- `np.dot`
- operador `@`
- `np.einsum`
- indexacao vetorizada
- `np.where`
- distribuicao normal com `np.random.randn`

Este projeto depende fortemente disso.

### Etapa 4: geometria e trigonometria

Voce precisa entender:

- seno e cosseno
- angulos em graus e radianos
- vetor tangente
- distancia entre pontos
- projecao em segmento

Isso aparece na fisica do carro e na pista.

### Etapa 5: redes neurais basicas

Estude:

- neuronio artificial
- pesos e vieses
- funcao de ativacao
- camada oculta
- forward pass
- normalizacao de entrada

Nao comece por transformers. Comece por redes pequenas como esta.

### Etapa 6: algoritmos geneticos

Estude:

- populacao
- fitness
- selecao
- elitismo
- crossover
- mutacao
- exploracao vs explotacao
- convergencia prematura

Esse projeto e quase um laboratorio disso.

### Etapa 7: simulacao e controle

Depois estude:

- agentes em ambiente
- sensores e atuadores
- funcao de recompensa
- dinamica de sistema simples

### Etapa 8: visualizacao

Aprenda a visualizar o processo. Ver o aprendizado ajuda muito mesmo.

Este projeto usa Matplotlib, mas voce tambem pode brincar com:

- Pygame
- Arcade
- Manim, para explicacoes

## 20. Como estudar usando este projeto

Uma boa estrategia:

1. leia primeiro [sim/track.py](/home/luk/neural-network-circuit-game/sim/track.py), porque ele define o mundo
2. depois leia [sim/neural_network.py](/home/luk/neural-network-circuit-game/sim/neural_network.py), porque ele define o cerebro
3. depois leia [sim/simulacao.py](/home/luk/neural-network-circuit-game/sim/simulacao.py), porque ele define o aprendizado
4. por fim leia [sim/visualizacao.py](/home/luk/neural-network-circuit-game/sim/visualizacao.py), porque ele mostra tudo acontecendo

Enquanto le isso, faca testes pequenos:

- diminua a populacao para entender mais facil
- aumente a mutacao para ver o caos
- desligue penalidades para ver o que degrada
- ligue sensores na visualizacao
- troque a arquitetura da rede

## 21. O que mais chama atencao tecnicamente neste projeto

Alguns pontos bons de engenharia:

- separacao entre motor da simulacao e visualizacao
- geometria da pista desacoplada da interface
- configuracao centralizada em JSON
- uso de SDF para consulta rapida de pista
- forward batch vetorizado para varios carros
- mutacao adaptativa conforme estagnacao
- editor de pista integrado ao fluxo do simulador

Ou seja, nao e so um experimento de IA. E tambem um projeto de software bem organizado para estudo.

## 22. Resposta curta para sua pergunta principal

### Ele fez uma rede neural?

Sim. Uma rede feedforward pequena implementada na mao com pesos, vieses e sigmoid.

### Como ele treinou?

Nao foi com backpropagation. Foi com algoritmo genetico: selecao, crossover e mutacao dos pesos.

### O que ele usou?

Principalmente Python, NumPy e Matplotlib.

### Isso e ML ou IA?

Os dois. Mais precisamente, neuroevolucao dentro de machine learning evolutivo.

### O que voce precisa estudar?

Python, NumPy, algebra linear basica, trigonometria, redes neurais simples, algoritmos geneticos e simulacoes de agentes.

## 23. Proximos experimentos que valem ouro para aprender

Se voce quiser aprender de verdade com este codigo, tente fazer estas mudancas:

1. trocar a arquitetura de `8 -> 14 -> 2` para `8 -> 10 -> 6 -> 2`
2. substituir sigmoid por `tanh` na camada oculta
3. testar mutacao por probabilidade em vez de mutar tudo com ruido
4. mudar a recompensa para premiar mais consistencia de volta do que velocidade
5. desenhar a trajetoria do melhor carro ao longo das geracoes
6. salvar e recarregar o melhor cerebro em arquivo separado
7. comparar essa abordagem com uma rede treinada por reinforcement learning de verdade

## 24. Fechamento

Se voce estava pensando "isso parece magica", a resposta curta e: nao e magica.

O projeto combina quatro blocos simples:

- percepcao por sensores
- uma rede neural pequena
- uma regra de pontuacao
- um algoritmo genetico que mexe nos pesos

O comportamento inteligente aparece da interacao entre esses blocos ao longo de muitas geracoes.

E exatamente por isso esse projeto e bom para estudar IA: ele deixa visivel o caminho entre numeros e comportamento.