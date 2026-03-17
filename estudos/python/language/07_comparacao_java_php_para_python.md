# Modulo Python 07: Comparacao Java/PHP para Python

Este modulo foi feito para voce que ja programa em outra stack e quer aprender Python sem confusao.

Objetivo: aproveitar o que voce ja sabe e ajustar o mindset.

## 1. O que muda de verdade

- Python usa menos boilerplate
- modulo (arquivo) e unidade muito importante
- tipagem e dinamica por padrao
- legibilidade e prioridade alta

## 2. Tabela de equivalencias

| Conceito | Java/PHP | Python |
|---|---|---|
| Classe | `class Car {}` | `class Carro:` |
| Metodo atual | `this` | `self` |
| Lista | `ArrayList`, `array` | `list` |
| Mapa | `HashMap`, array associativo | `dict` |
| Ordenar | `Collections.sort`, `usort` | `sorted`, `.sort()` |
| Loop com indice | `for (int i=0...)` | `for i, x in enumerate(...)` |
| Try/catch | `try/catch` | `try/except` |
| Pacote/modulo | pacote/classe | arquivo `.py` + pasta |

## 3. Ajuste mental numero 1: modulo importa muito

Em Java, muita coisa gira em torno de classe.
Em Python, modulo tem vida propria.

Voce pode ter:

- funcoes utilitarias em modulo
- constantes em modulo
- classes em modulo

Sem problema.

## 4. Ajuste mental numero 2: tipagem dinamica com disciplina

Python nao exige tipo em toda variavel, mas isso nao significa bagunca.

Disciplina pratica:

- nomes bons
- funcoes pequenas
- validacao de entrada
- type hints quando ajudar leitura

## 5. Ajuste mental numero 3: loops mais declarativos

Evite escrever Python como Java:

```python
for i in range(len(lista)):
    print(i, lista[i])
```

Prefira:

```python
for i, item in enumerate(lista):
    print(i, item)
```

## 6. Ajuste mental numero 4: listas e dicionarios sao mutaveis

Mutabilidade e fonte de poder e bugs.

```python
a = [1, 2]
b = a
b.append(3)
print(a)  # [1, 2, 3]
```

## 7. Classes: quando usar e quando nao usar

Use classe quando ha estado e comportamento juntos.

Exemplo no projeto:

- carro
- rede neural
- simulador

Nao use classe para tudo. Funcoes puras tambem sao excelentes.

## 8. Erros comuns de quem vem de Java/PHP

- criar getters/setters para tudo sem necessidade
- usar heranca onde composicao bastava
- ignorar built-ins Python e reimplementar roda
- capturar excecao generica e esconder bug

## 9. Pythonico no dia a dia

- `enumerate` para indice + item
- `zip` para iterar em paralelo
- `sorted(..., key=...)` para ordenar com criterio
- `with open(...)` para arquivo

## 10. Nao compare performance sem contexto

Python puro em loop pesado pode ser mais lento que Java.
Mas com NumPy vetorizado, muito trabalho pesado roda em C por baixo.

No contexto deste projeto, isso faz grande diferenca.

## 11. Miniguia de traducao mental

- "quero utilitario estatico" -> funcao em modulo
- "quero enum" -> pode usar `Enum`, mas nem sempre precisa
- "quero DTO" -> `dataclass` pode ajudar
- "quero stream map/filter" -> list comprehension ou `map/filter`

## 12. Checklist de migracao de stack

- estou tentando forcar padrao da linguagem antiga?
- usei built-ins Python antes de escrever codigo manual?
- separei responsabilidades por modulo?
- tratei erros nas bordas (arquivo/rede/JSON)?

## Exercicios

### Exercicio 1

Pegue um trecho Java/PHP seu e reescreva em Python idiomatico com:

- `enumerate`
- `sorted(..., key=...)`
- `with open(...)`

### Exercicio 2

Escolha uma classe do projeto e diga por que ela deveria continuar classe (e nao virar funcoes soltas).

### Exercicio 3

Escreva tres "vicios" da sua linguagem de origem que voce quer evitar no Python.
