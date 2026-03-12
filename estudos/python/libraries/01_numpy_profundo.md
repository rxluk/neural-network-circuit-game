# Modulo Bibliotecas 01: NumPy profundo para este projeto

NumPy e provavelmente a biblioteca mais importante do projeto inteiro.

## 1. O papel do NumPy aqui

Ele e usado para:

- representar sensores e pesos
- fazer forward pass da rede neural
- calcular distancias geometricas
- processar varios carros em lote
- gerar ruido aleatorio para mutacao

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

## 3. `np.array`

Base de tudo. Transforma listas e sequencias em arrays numericos.

## 4. `np.zeros`

Usado no projeto para inicializar vieses e buffers.

```python
b1 = np.zeros(14)
```

## 5. `np.random.randn`

Usado para:

- pesos iniciais
- mutacao

Ele amostra da distribuicao normal padrao.

## 6. `np.clip`

No projeto, protege a sigmoid de overflow numerico antes de `exp`.

Isso e um detalhe importante de robustez numerica.

## 7. `np.stack`

Empilha arrays para criar um batch.

No forward batch, varias matrizes de varias redes sao empilhadas para processar tudo junto.

## 8. `np.einsum`

Muito poderoso para operacoes tensorais. No projeto ele faz o forward em lote de varias redes ao mesmo tempo.

Se voce nao domina ainda, tudo bem. Aprenda primeiro `@`, depois volte.

## 9. `np.argmin` e `np.argmax`

Usados para encontrar:

- indice do ponto mais proximo na centerline
- primeiro hit de sensor
- melhores individuos

## 10. `np.linspace` e `np.arange`

### `np.linspace`

Bom para gerar amostras uniformes entre dois extremos.

### `np.arange`

Bom para sequencias com passo fixo.

No projeto, ambos aparecem em geometria e sensores.

## 11. `np.outer`

Usado na construcao da spline Catmull-Rom.

Esse ponto mostra que NumPy aqui nao serve so para rede neural. Ele tambem serve para modelagem geometrica.

## 12. `np.hypot`

Forma conveniente de calcular magnitudes em 2D.

## 13. Pensar em shapes

Se voce quiser dominar NumPy, sua pergunta constante precisa ser:

- qual e o shape desta estrutura antes e depois da operacao?

## 14. Fluxos NumPy do projeto que valem estudar no codigo

- sensores em lote
- SDF e distancia da pista
- forward batch
- mutacao dos pesos
- calculo de progresso na centerline

## 15. O que estudar para nivel forte

- broadcasting
- indexacao booleana
- algebra linear com `@`
- vetorizacao
- estabilidade numerica

## Exercicios

### Exercicio 1

Implemente uma camada `x @ W + b` e imprima shapes em cada etapa.

### Exercicio 2

Crie 20 redes pequenas e um batch de entradas, e tente reproduzir um forward em lote com `np.stack` e `np.einsum`.
