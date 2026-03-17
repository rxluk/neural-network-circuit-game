# Modulo Python 06: Excecoes, tratamento de erros e context manager

Este modulo existe para resolver um problema real: programas quebram.

Programador maduro nao e quem nunca erra.
Programador maduro e quem trata erro de forma clara e previsivel.

## 1. O que e uma excecao

Excecao e um erro em tempo de execucao.

Exemplos comuns:

- arquivo nao encontrado
- chave ausente em dicionario
- divisao por zero
- JSON invalido

## 2. Estrutura basica: `try/except`

```python
try:
    valor = int("abc")
except ValueError:
    print("valor invalido para inteiro")
```

Leitura simples:

- `try`: tenta executar
- `except`: captura erro especifico

## 3. `else` e `finally`

```python
try:
    numero = int("42")
except ValueError:
    print("nao deu para converter")
else:
    print("conversao ok:", numero)
finally:
    print("sempre executa")
```

Use `finally` para limpeza de recurso quando necessario.

## 4. Capture erros especificos

Evite:

```python
except Exception:
    pass
```

Isso esconde problemas.

Prefira:

```python
except FileNotFoundError:
    ...
except json.JSONDecodeError:
    ...
```

## 5. Exemplo real com JSON de configuracao

```python
import json


def carregar_config(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"arquivo nao encontrado: {caminho}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"json invalido em {caminho}: {exc}")
```

## 6. O que e context manager (`with`)

Context manager e um objeto que sabe:

- como abrir/recurso iniciar (`__enter__`)
- como fechar/recurso limpar (`__exit__`)

`open(...)` ja implementa isso.

```python
with open("config.json", "r", encoding="utf-8") as f:
    dados = f.read()
```

Ao sair do bloco, o arquivo e fechado automaticamente.

## 7. Criando seu proprio context manager

```python
class Cronometro:
    def __enter__(self):
        import time
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        import time
        duracao = time.perf_counter() - self._inicio
        print(f"duracao: {duracao:.4f}s")
```

Uso:

```python
with Cronometro():
    soma = sum(range(1_000_000))
```

## 8. `raise`: disparando erros com intencao

```python
def validar_sensores(sensores):
    if len(sensores) != 7:
        raise ValueError("esperado vetor de 7 sensores")
```

Voce dispara excecao quando entrada invalida colocaria o sistema em estado errado.

## 9. Excecoes customizadas

```python
class ErroDeConfiguracao(Exception):
    pass
```

```python
def validar_config(config):
    if "taxa_mutacao" not in config:
        raise ErroDeConfiguracao("campo taxa_mutacao ausente")
```

## 10. Ponte para Java/PHP

- Java: `try/catch/finally`
- Python: `try/except/else/finally`

- Java checked exception: Python nao tem esse mecanismo por tipo verificado em compilacao
- PHP: `try/catch/finally` parecido, mas em Python os tipos de erro mais comuns mudam

## 11. Boas praticas de projeto

- trate erro perto da borda (I/O, arquivo, rede)
- valide entrada cedo
- nunca silencie erro sem log
- mensagem de erro precisa ajudar quem vai debugar

## 12. Checklist rapido

- estou capturando excecao especifica?
- `with` esta sendo usado para recursos?
- erro tem mensagem clara?
- estou mascarando bug sem querer?

## Exercicios

### Exercicio 1

Implemente uma funcao que abre um arquivo JSON de pista e trata:

- arquivo inexistente
- JSON invalido
- campo obrigatorio ausente

### Exercicio 2

Crie um context manager chamado `LogPasso` que imprime:

- "inicio"
- "fim"
- tempo total

### Exercicio 3

Escreva um trecho com `try/except/else/finally` e explique em uma frase o papel de cada bloco.
