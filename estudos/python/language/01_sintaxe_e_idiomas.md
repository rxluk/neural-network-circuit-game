# Modulo Python 01: Sintaxe e Idiomas de Python para este dominio

Este modulo e sobre uma diferenca importante: saber sintaxe nao e o mesmo que escrever Python bem.

## 1. O que significa Python idiomatico

Significa usar a linguagem do jeito que ela foi pensada para ser usada:

- com clareza
- com concisao razoavel
- sem rigidez desnecessaria
- aproveitando as estruturas nativas e a standard library

## 2. Atribuicao multipla e desempacotamento

Python torna natural manipular coordenadas e pares.

```python
x, y = 10.0, 5.0
posicao = (x, y)
px, py = posicao
```

Isso aparece muito em simulacao geometrica.

## 3. List comprehensions

Muito uteis quando voce quer construir colecoes derivadas de outras.

```python
vivos = [c for c in carrinhos if c.vivo]
scores = [c.pontos_acumulados for c in carrinhos]
```

### Quando nao usar

Se a regra ficar longa demais, troque por um loop claro.

## 4. `enumerate`, `zip` e `sorted`

### `enumerate`

```python
for i, carro in enumerate(carrinhos):
    print(i, carro.vivo)
```

### `zip`

```python
for angulo, distancia in zip(angulos, distancias):
    print(angulo, distancia)
```

### `sorted`

```python
melhores = sorted(carrinhos, key=lambda c: c.pontos_acumulados, reverse=True)
```

Esses tres aparecem o tempo todo em codigo de simulacao e ML.

## 5. Verdade de objetos e checagens simples

Python permite escrever condicoes de forma limpa:

```python
if carrinhos:
    print("ha carros")
```

Em vez de:

```python
if len(carrinhos) > 0:
    print("ha carros")
```

## 6. Acesso a atributos em objetos

No projeto, o estilo principal e orientado a atributos:

```python
carro.velocidade
carro.vivo
carro.pontos_acumulados
```

Isso deixa o modelo mental muito direto.

## 7. Mutabilidade: ponto critico em Python

Listas, dicionarios e arrays sao mutaveis. Isso e poderoso e perigoso.

### Exemplo de perigo

Se dois objetos compartilham a mesma estrutura mutavel por engano, uma alteracao em um pode afetar o outro.

Por isso o projeto copia pesos com `.copy()` ao clonar redes.

## 8. `copy` conceitual no projeto

Quando voce copia uma rede neural, precisa copiar os dados, nao apenas a referencia.

Caso contrario, mutar uma rede pode estragar outra.

## 9. `with open(...)`

Padrao correto para arquivos:

```python
with open("config.json", "r", encoding="utf-8") as f:
    dados = json.load(f)
```

Motivo:

- fecha o arquivo corretamente
- e o estilo Python esperado

## 10. Compreendendo `self`

Python nao tem palavra magica escondida. `self` e apenas a convencao para se referir ao objeto atual.

```python
class Carro:
    def acelerar(self):
        self.velocidade += 1
```

## 11. Python e dinamico

Voce nao declara tipos de forma obrigatoria como em Java ou C#.

Isso acelera prototipagem, mas exige disciplina mental.

## 12. Como escrever melhor Python em projeto tecnico

- prefira nomes semanticos
- mantenha funcoes com responsabilidade definida
- isole regras numericas em funcoes/metodos especificos
- nao misture rendering com fisica se puder evitar
- coloque configuracao fora do codigo quando fizer sentido

## 13. Erros de estilo comuns

### Escrever Python como se fosse outra linguagem

Exemplo:

- usar loops verbosos quando uma estrutura idiomatica e mais clara
- empacotar tudo em classes desnecessarias

### Compactar demais

Codigo Python conciso demais tambem pode virar enigma.

O alvo nao e o menor codigo. E o mais legivel.

## 14. Mini-catalogo de coisas que voce vai ver muito

```python
max(lista)
min(lista)
sum(lista)
any(condicoes)
all(condicoes)
range(n)
```

E no projeto numerico:

```python
np.mean(...)
np.argmax(...)
np.argmin(...)
```

## 15. Um olhar maduro

Saber Python para este contexto significa conseguir desenhar estruturas que representem bem o problema. Nao so lembrar sintaxe.

## Exercicios

### Exercicio 1

Reescreva um loop seu recente usando `enumerate`, `zip` ou list comprehension de forma mais legivel.

### Exercicio 2

Implemente uma classe `Rede` com atributos `W1`, `b1`, `W2`, `b2` e um metodo `copiar_de()` que use copia real dos arrays, nao referencia compartilhada.
