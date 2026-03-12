# Modulo Python 00: Mapa do Python que este projeto usa

Quando se fala em aprender Python para IA, muita gente cai em dois extremos ruins:

- estudar so sintaxe e nunca construir nada
- pular direto para biblioteca e copiar codigo sem entender a linguagem

O que voce precisa aqui e um mapa do Python realmente usado neste projeto.

## 1. Quais aspectos de Python aparecem aqui

Este projeto usa Python de forma bastante classica e limpa. Os pilares sao:

- modulos
- classes
- metodos
- dicionarios de configuracao
- arrays NumPy
- funcoes auxiliares
- leitura de arquivos JSON
- eventos de interface com Matplotlib

## 2. O que este projeto nao exige fortemente da linguagem

Voce nao precisa dominar agora:

- metaclasses
- decoradores avancados
- async/await
- descriptors
- context managers sofisticados
- type system avancado

Isso e importante para tirar ansiedade. O projeto e tecnicamente rico, mas a linguagem usada nele e acessivel.

## 3. O estilo arquitetural dele

O projeto tem uma divisao muito saudavel:

- um modulo para rede neural
- um modulo para pista e fisica
- um modulo para simulacao e evolucao
- um modulo para visualizacao

Esse tipo de separacao e uma habilidade Python importante por si so.

## 4. O que voce precisa prestar atencao ao ler o codigo

Quando abrir os arquivos, pergunte:

- este codigo representa estado ou comportamento?
- esta funcao calcula algo ou altera algo?
- este metodo pertence ao carro, ao simulador ou a rede?
- este valor deveria estar em JSON ou hardcoded?

## 5. O Python aqui e orientado a objetos ou funcional?

Predominantemente orientado a objetos, com apoio de funcoes.

Exemplos:

- `CarrinhoIA` guarda estado do carro
- `RedeNeuralCarrinho` guarda pesos e comportamento da rede
- `SimuladorBase` organiza o treinamento

Mas tambem ha funcoes puras e utilitarias, como geracao de geometria e carregamento de recursos.

## 6. O jeito Python de fazer as coisas aqui

O projeto usa varios habitos tipicos de Python bem aplicado:

- atributos claros em objetos
- nomes legiveis
- dicionarios para configuracao
- modulos pequenos com responsabilidade clara
- uso forte de bibliotecas em vez de reinventar tudo

## 7. O que estudar na linguagem para acompanhar este projeto bem

### Essencial

- atribuicao e tipos dinamicos
- listas, tuplas, dicionarios e conjuntos
- `for`, `if`, `return`
- funcoes e metodos
- classes e `self`
- imports
- leitura e escrita de arquivos

### Muito util

- list comprehensions
- `enumerate`
- `zip`
- slicing
- desempacotamento
- `with open(...)`

## 8. Erro comum ao estudar Python para IA

Ficar preso em teoria de linguagem e nao conectar com modelagem.

O que faz Python ficar vivo aqui nao e a sintaxe isolada. E o uso da linguagem para modelar:

- um carro
- uma pista
- um cerebro
- um processo evolutivo

## 9. Meta deste subbloco

Ao terminar esta pasta `language/`, voce deve conseguir:

- reestruturar um projeto parecido
- ler classes e metodos sem se perder
- escrever Python mais natural para este dominio

## Exercicios

### Exercicio 1

Abra `sim/track.py` e diga quais partes sao estado, quais sao funcoes utilitarias e quais sao regras de comportamento.

### Exercicio 2

Escreva um mini-mapa do projeto atual com 5 caixas: entrada, ambiente, agente, treino e visualizacao.
