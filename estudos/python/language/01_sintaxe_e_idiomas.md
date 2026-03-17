# Modulo Python 01: Sintaxe e Idiomas de Python para este dominio

Este modulo tem um objetivo claro: transformar voce em alguem que escreve Python com clareza.

Se voce vem de Java, PHP, C# ou JavaScript, pense assim:

- sua logica ja existe
- seu desafio agora e falar "o idioma Python"

## 1. O que e Python idiomatico

Escrever Python idiomatico e usar as ferramentas da linguagem de forma natural:

- codigo legivel primeiro
- menos boilerplate
- estruturas nativas (`list`, `dict`, `set`, `tuple`)
- funcoes built-in (`enumerate`, `zip`, `sorted`, `any`, `all`)

Nao e sobre escrever "menos linhas a qualquer custo".
E sobre escrever codigo que outra pessoa entende em poucos segundos.

## 2. Comparacao rapida para quem vem de Java/PHP

- Java: muito `for (int i = 0; i < n; i++)`
- Python: quase sempre `for item in colecao`

- Java: classes para quase tudo
- Python: classes quando ha estado + comportamento; funcoes quando basta transformar dados

- PHP: arrays sao usados para tudo
- Python: lista, dicionario, tupla e set tem papeis diferentes

## 3. Atribuicao multipla e desempacotamento

```python
x, y = 10.0, 5.0
posicao = (x, y)
px, py = posicao
```

Quando usar:

- coordenadas (`x, y`)
- retorno de funcoes com mais de um valor
- troca de variaveis (`a, b = b, a`)

## 4. List comprehensions

```python
vivos = [c for c in carrinhos if c.vivo]
scores = [c.pontos_acumulados for c in carrinhos]
```

Use quando:

- a transformacao e curta e direta

Evite quando:

- a regra tem muitos `if/else`
- a leitura fica cansativa

Nesses casos, um loop normal e melhor.

## 5. `enumerate`: iterar com indice sem dor

### O que faz

`enumerate` percorre um iteravel e entrega dois valores por vez:

- indice
- elemento

### Assinatura

```python
enumerate(iterable, start=0)
```

### Retorno

Retorna um iterador (lazy), nao uma lista pronta.

### Exemplo basico

```python
for i, carro in enumerate(carrinhos):
    print(i, carro.vivo)
```

### Exemplo iniciando em 1

```python
for posicao, nome in enumerate(["ana", "bia", "caio"], start=1):
    print(posicao, nome)
```

### Quando usar

- sempre que voce precisa do item e da posicao
- para logs e mensagens de depuracao

### Armadilhas

- `enumerate(None)` gera erro
- `list(enumerate(...))` materializa tudo em memoria; para listas gigantes, evite se nao precisar

### Equivalente mental Java/PHP

- Java: `for (int i = 0; i < lista.size(); i++)`
- Python idiomatico: `for i, item in enumerate(lista)`

## 6. `zip`: caminhar em paralelo

### O que faz

`zip` junta elementos de varios iteraveis por posicao.

### Assinatura

```python
zip(*iterables)
```

### Retorno

Retorna um iterador de tuplas (lazy).

### Exemplo basico

```python
for angulo, distancia in zip(angulos, distancias):
    print(angulo, distancia)
```

### Exemplo com 3 listas

```python
for nome, score, vivo in zip(nomes, scores, vivos):
    print(nome, score, vivo)
```

### Regra importante

`zip` para no menor iteravel.

```python
list(zip([1, 2, 3], [10, 20]))  # [(1, 10), (2, 20)]
```

### Quando usar

- dados alinhados por posicao
- sensores e angulos
- nomes e valores de metricas

### Armadilhas

- tamanhos diferentes podem esconder dados que ficaram de fora
- para validar, compare `len(...)` antes do zip quando necessario

### Equivalente mental Java/PHP

- Java nao tem `zip` nativo simples em colecoes comuns
- em Python, isso ja vem pronto e melhora legibilidade

## 7. `sorted`: ordenar sem alterar original

### O que faz

Cria uma nova lista ordenada a partir de qualquer iteravel.

### Assinatura

```python
sorted(iterable, key=None, reverse=False)
```

### Retorno

Retorna uma nova lista (eager).

### Exemplo basico

```python
melhores = sorted(carrinhos, key=lambda c: c.pontos_acumulados, reverse=True)
```

### Exemplo com multiplas chaves

```python
ordenados = sorted(carrinhos, key=lambda c: (-c.pontos_acumulados, c.tempo_vida))
```

### `sorted` vs `.sort()`

- `sorted(lista)` retorna nova lista
- `lista.sort()` altera a propria lista

### Quando usar

- ranking de individuos
- gerar relatorios ordenados sem mexer no original

### Armadilhas

- `key` deve ser funcao barata se usado em dados muito grandes
- objetos sem comparacao clara exigem `key`

### Equivalente mental Java/PHP

- Java: `stream().sorted(...)` ou `Collections.sort(...)`
- PHP: `usort(...)`

## 8. Verdade de objetos (`truthy` e `falsy`)

```python
if carrinhos:
    print("ha carros")
```

Mais idiomatico que:

```python
if len(carrinhos) > 0:
    print("ha carros")
```

Valores comuns considerados falsos:

- `None`
- `0`
- `""`
- `[]`, `{}`, `set()`

## 9. Mutabilidade: bug classico em Python

Listas e dicionarios sao mutaveis.

### Erro comum

```python
linhas = [[]] * 3
linhas[0].append(1)
print(linhas)  # [[1], [1], [1]]
```

### Forma correta

```python
linhas = [[] for _ in range(3)]
linhas[0].append(1)
print(linhas)  # [[1], [], []]
```

No projeto, isso importa muito ao copiar pesos de rede.

## 10. `with open(...)`: padrao para arquivos

```python
import json

with open("config.json", "r", encoding="utf-8") as f:
    dados = json.load(f)
```

Vantagens:

- fecha arquivo automaticamente
- reduz risco de vazamento de recurso
- padrao esperado em Python profissional

## 11. `self` sem misterio

```python
class Carro:
    def acelerar(self):
        self.velocidade += 1
```

`self` e a referencia ao objeto atual.
Pense como `this` em Java/C#.

## 12. Mini-catalogo de built-ins que aparecem sempre

```python
max(lista)
min(lista)
sum(lista)
any(condicoes)
all(condicoes)
range(n)
```

E no trecho numerico:

```python
np.mean(...)
np.argmax(...)
np.argmin(...)
```

## 13. Checklist rapido de qualidade Python

- nomes explicam intencao?
- loops podem usar `enumerate` ou `zip`?
- ordenacao usa `sorted(..., key=...)` claro?
- ha risco de mutabilidade compartilhada?
- arquivo/JSON esta com `with open`?

## Exercicios

### Exercicio 1

Pegue um loop antigo seu e reescreva com `enumerate`, `zip` ou list comprehension.
Depois compare qual versao ficou mais legivel.

### Exercicio 2

Implemente uma classe `Rede` com `W1`, `b1`, `W2`, `b2` e um metodo `copiar_de()` que faz copia real (`.copy()`) em vez de compartilhar referencia.

### Exercicio 3

Crie 3 exemplos pequenos mostrando:

- `sorted` sem alterar lista original
- `zip` truncando no menor iteravel
- `enumerate(..., start=1)` para ranking humano
