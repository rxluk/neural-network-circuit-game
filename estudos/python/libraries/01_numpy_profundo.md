# Modulo Bibliotecas 01: NumPy profundo para este projeto

NumPy e provavelmente a biblioteca mais importante do projeto inteiro.

Se Python puro e uma calculadora comum, NumPy e uma calculadora cientifica turbo.

## Regra de ouro deste modulo

Antes de qualquer operacao, pergunte:

- qual e o shape de entrada?
- qual shape eu espero na saida?

## 1. O papel do NumPy aqui

Ele e usado para:

- representar sensores e pesos
- fazer forward pass da rede neural
- calcular distancias geometricas
- processar varios carros em lote
- gerar ruido aleatorio para mutacao

Sem NumPy, o projeto ficaria muito mais lento e mais complicado.

## 2. Objetos principais que voce precisa dominar

- `np.array`
- `np.zeros`
- `np.random.randn`
- `np.clip`
- `np.stack`
- `np.where`
- `np.einsum`
- `np.argmin`
- `np.argmax`
- `np.mean`
- `np.hypot`
- `np.outer`
- `np.linspace`
- `np.arange`

Voce nao precisa decorar tudo no primeiro dia.
Comece pelos 7 primeiros e avance gradualmente.

## 3. `np.array`

Base de tudo. Transforma listas e sequencias em arrays numericos.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x.shape)  # (3,)
```

## 4. `np.zeros`

Usado no projeto para inicializar vieses e buffers.

```python
b1 = np.zeros(14)
```

Interprete assim: "crie um vetor de 14 zeros".

## 5. `np.random.randn`

Usado para:

- pesos iniciais
- mutacao

Ele amostra da distribuicao normal padrao.

```python
W = np.random.randn(8, 14)
print(W.shape)  # (8, 14)
```

## 6. `np.clip`

No projeto, protege a sigmoid de overflow numerico antes de `exp`.

Isso e um detalhe importante de robustez numerica.

```python
z = np.array([-1000.0, 0.0, 1000.0])
z_seguro = np.clip(z, -60.0, 60.0)
```

## 7. `np.stack`

Empilha arrays para criar um batch.

No forward batch, varias matrizes de varias redes sao empilhadas para processar tudo junto.

```python
a = np.ones((8, 14))
b = np.zeros((8, 14))
pilha = np.stack([a, b], axis=0)
print(pilha.shape)  # (2, 8, 14)
```

## 8. `np.einsum`

Muito poderoso para operacoes tensorais. No projeto ele faz o forward em lote de varias redes ao mesmo tempo.

Se voce nao domina ainda, tudo bem. Aprenda primeiro `@`, depois volte.

Exemplo mental:

- com `@` voce multiplica uma matriz por outra
- com `einsum` voce descreve "como os eixos se combinam"

## 8.1 Ponte pratica: `@` antes de `einsum`

```python
import numpy as np

x = np.random.randn(32, 8)      # batch de 32 entradas
W = np.random.randn(8, 14)
b = np.random.randn(14)

z = x @ W + b                   # (32, 14)
```

Mesmo calculo com `einsum`:

```python
z2 = np.einsum("bi,ij->bj", x, W) + b
```

## 9. `np.argmin` e `np.argmax`

## 9. `np.argmin` e `np.argmax`

Usados para encontrar:

- indice do ponto mais proximo na centerline
- primeiro hit de sensor
- melhores individuos

```python
scores = np.array([10, 50, 30])
melhor_idx = np.argmax(scores)  # 1
```

## 10. `np.linspace` e `np.arange`

### `np.linspace`

Bom para gerar amostras uniformes entre dois extremos.

### `np.arange`

Bom para sequencias com passo fixo.

No projeto, ambos aparecem em geometria e sensores.

```python
print(np.linspace(0.0, 1.0, 5))  # [0.   0.25 0.5  0.75 1.  ]
print(np.arange(0, 5, 2))        # [0 2 4]
```

## 11. `np.outer`

Usado na construcao da spline Catmull-Rom.

Esse ponto mostra que NumPy aqui nao serve so para rede neural. Ele tambem serve para modelagem geometrica.

```python
a = np.array([1, 2, 3])
b = np.array([10, 20])
print(np.outer(a, b))
```

## 12. `np.hypot`

Forma conveniente de calcular magnitudes em 2D.

```python
dx = np.array([3.0, 5.0])
dy = np.array([4.0, 12.0])
dist = np.hypot(dx, dy)  # [5.0, 13.0]
```

## 13. Pensar em shapes

Se voce quiser dominar NumPy, sua pergunta constante precisa ser:

- qual e o shape desta estrutura antes e depois da operacao?

## 13.1 Broadcasting explicado como gente grande

Broadcasting e quando NumPy "expande" dimensoes compativeis sem copiar dados de verdade.

Regras essenciais (da direita para esquerda):

- dimensoes iguais: ok
- uma das dimensoes e 1: ok (expande)
- caso contrario: erro

Exemplos:

```python
import numpy as np

a = np.ones((2, 5))
b = np.arange(5)        # shape (5,)
print((a + b).shape)    # (2, 5)
```

```python
a = np.ones((2, 5))
b = np.ones((3, 5))
# a + b -> erro: 2 e 3 nao sao compativeis
```

Diagrama mental simples:

```text
(2, 5)
(1, 5)  -> expande para (2, 5) e soma
```

## 13.2 Erros classicos de shape

- confundir vetor linha `(1, n)` com vetor simples `(n,)`
- esquecer de manter dimensao de batch
- somar bias com shape incorreto

Ferramenta pratica:

```python
print("x", x.shape, "W", W.shape, "b", b.shape)
```

## 14. Fluxos NumPy do projeto que valem estudar no codigo

- sensores em lote
- SDF e distancia da pista
- forward batch
- mutacao dos pesos
- calculo de progresso na centerline

## 15. Guia "formula -> codigo"

Formula de camada densa:

$$
z = xW + b
$$

NumPy:

```python
z = x @ W + b
```

Com ativacao sigmoid:

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

```python
z_seguro = np.clip(z, -60.0, 60.0)
a = 1.0 / (1.0 + np.exp(-z_seguro))
```

## 16. O que estudar para nivel forte

- broadcasting
- indexacao booleana
- algebra linear com `@`
- vetorizacao
- estabilidade numerica

## 17. Checklist de dominio NumPy para este projeto

- consigo prever shapes sem executar?
- sei usar `@` com confianca?
- entendo quando `broadcasting` funciona?
- sei evitar overflow em `exp`?
- consigo montar um forward batch?

## Exercicios

### Exercicio 1

Implemente uma camada `x @ W + b` e imprima shapes em cada etapa.

### Exercicio 2

Crie 20 redes pequenas e um batch de entradas, e tente reproduzir um forward em lote com `np.stack` e `np.einsum`.

### Exercicio 3

Sem rodar o codigo, diga se funciona ou falha (e por que):

- `(32, 8) @ (8, 14)`
- `(32, 8) + (14,)`
- `(32, 14) + (14,)`
- `(32, 14) + (32,)`
