# Modulo Python 08: Slicing, generators e funcoes funcionais

Este modulo fecha um gap importante: recursos que aparecem no ecossistema Python e confundem muita gente.

## 1. Slicing: recortar sequencias com precisao

Formato geral:

```python
sequencia[inicio:fim:passo]
```

Regras:

- `inicio` incluido
- `fim` excluido
- `passo` opcional (padrao 1)

Exemplos:

```python
nums = [0, 1, 2, 3, 4, 5, 6]
print(nums[1:4])   # [1, 2, 3]
print(nums[:3])    # [0, 1, 2]
print(nums[3:])    # [3, 4, 5, 6]
print(nums[::2])   # [0, 2, 4, 6]
print(nums[::-1])  # [6, 5, 4, 3, 2, 1, 0]
```

## 2. Slicing com NumPy

```python
import numpy as np

M = np.arange(12).reshape(3, 4)
print(M[:, 1:3])
```

Leitura:

- todas as linhas
- colunas de indice 1 ate 2

## 3. Generator: produzir valores sob demanda

Generator e uma forma de gerar valores aos poucos.

Vantagem:

- menos memoria
- bom para fluxos grandes

```python
def contar_ate(n):
    i = 0
    while i < n:
        yield i
        i += 1
```

Uso:

```python
for x in contar_ate(5):
    print(x)
```

## 4. `yield` vs `return`

- `return`: encerra a funcao
- `yield`: pausa e continua depois

Pense em `yield` como "entrega uma unidade por vez".

## 5. Generator expression

```python
quadrados = (x * x for x in range(10))
```

Parecido com list comprehension, mas lazy.

```python
quadrados_lista = [x * x for x in range(10)]
```

## 6. `map` e `filter`

### `map`

Aplica funcao em cada item.

```python
nums = [1, 2, 3]
dobro = map(lambda x: x * 2, nums)
print(list(dobro))  # [2, 4, 6]
```

### `filter`

Mantem itens que passam no teste.

```python
nums = [1, 2, 3, 4]
pares = filter(lambda x: x % 2 == 0, nums)
print(list(pares))  # [2, 4]
```

## 7. Quando preferir list comprehension

Em Python moderno, para casos simples:

```python
dobro = [x * 2 for x in nums]
pares = [x for x in nums if x % 2 == 0]
```

Isso costuma ficar mais legivel.

## 8. `reduce`

`reduce` acumula uma colecao em um valor final.

```python
from functools import reduce

nums = [1, 2, 3, 4]
soma = reduce(lambda acc, x: acc + x, nums, 0)
print(soma)  # 10
```

Para soma simples, prefira `sum(nums)`.

## 9. `sorted` com funcoes como chave

```python
carros = [
    {"nome": "A", "score": 10},
    {"nome": "B", "score": 30},
    {"nome": "C", "score": 20},
]

ordenados = sorted(carros, key=lambda c: c["score"], reverse=True)
```

## 10. Armadilhas comuns

- esquecer que `map/filter` retornam iteradores
- consumir iterador uma vez e tentar usar de novo
- slicing com indices errados por confundir fim inclusivo
- usar `reduce` onde `sum`, `max`, `min` seriam mais claros

## 11. Ponte para o projeto de rede neural

- slicing: extrair faixas de sensores
- generators: stream de episodios ou logs
- `sorted`: ranking de individuos por fitness
- `map/filter`: transformacoes simples em colecoes

## 12. Checklist de uso consciente

- preciso de lazy evaluation?
- legibilidade esta melhor ou pior?
- posso usar built-in mais simples?
- shape/indice esta correto no slicing?

## Exercicios

### Exercicio 1

Dado um vetor de sensores com 7 valores, pegue:

- os 3 primeiros
- os 3 ultimos
- apenas sensores de indice par

### Exercicio 2

Crie um generator que produza ids de geracao de 1 ate 100.

### Exercicio 3

Reescreva um `map/filter` com list comprehension e compare legibilidade.
