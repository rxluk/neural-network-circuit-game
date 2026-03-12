# Modulo IA Aplicacoes 01: Sensores, fitness e reward design

Se existe uma parte deste projeto que mais ensina engenharia de IA aplicada, e esta.

## 1. Sensores sao a interface entre mundo e agente

O agente nao recebe a pista inteira como onisciencia. Ele recebe um recorte do mundo em forma numerica.

No projeto, isso acontece por sensores de distancia e velocidade.

## 2. Escolher sensores e escolher a percepcao do agente

Isso e muito importante.

Se os sensores sao pobres demais, a rede nao tem informacao suficiente.

Se sao ricos demais, o sistema pode ficar pesado ou desnecessariamente complexo.

## 3. O projeto escolheu uma representacao boa porque

- os sensores cobrem frente e laterais
- a velocidade complementa o estado
- a entrada continua pequena o bastante para uma MLP simples

## 4. O que e fitness na pratica

Fitness e a traduçao numerica da pergunta:

- quao bom foi o comportamento deste agente?

## 5. Reward design e talvez a parte mais dificil

Porque voce precisa traduzir um objetivo intuitivo em sinais numericos que realmente empurrem a populacao na direcao certa.

## 6. O que este projeto premia bem

- progresso real na pista
- completar voltas
- velocidade util
- permanencia em regioes boas da pista

## 7. O que ele pune bem

- sair da pista
- andar na direcao errada
- ficar muito lento depois que ja deveria estar performando
- permanecer sem progresso

## 8. Reward hacking

Sempre que voce desenha reward, existe o risco de o agente aprender um atalho ruim que maximiza a pontuacao sem cumprir a intencao real.

Exemplo conceitual:

- se tempo vivo fosse muito recompensado e progresso pouco, o carro poderia aprender a enrolar

## 9. Como desenhar reward melhor

- comece simples
- observe comportamento real
- remova brechas
- adicione termos apenas quando houver motivo observavel

## 10. Sinal de reward mal desenhada

- fitness sobe, mas o comportamento parece pior
- agentes exploram glitches
- aprendizado estagna em estrategia estranha

## 11. Licao principal

Modelo ruim com reward boa pode te ensinar bastante. Modelo bom com reward ruim pode te enganar por muito tempo.

## Exercicios

### Exercicio 1

Escreva uma reward simples para um carro em corredor reto usando apenas 3 termos: progresso, colisao e velocidade.

### Exercicio 2

Descreva 2 exemplos de reward hacking que poderiam acontecer nesse tipo de simulacao e diga como voce mitigaria cada um.
