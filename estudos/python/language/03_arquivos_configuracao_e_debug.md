# Modulo Python 03: Arquivos, configuracao e debug de projeto tecnico

Projetos tecnicos bons nao sao so algoritmo. Eles tambem dependem de configuracao e observabilidade.

## 1. Por que JSON importa aqui

Neste projeto, `config.json` e `pista.json` tiram muitos detalhes do codigo-fonte.

Isso e excelente porque permite:

- testar hiperparametros rapidamente
- modificar pista sem reescrever logica
- separar dados de comportamento

## 2. Ler JSON em Python

```python
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
```

`config` vira um dicionario Python.

## 3. Escrita de JSON

```python
with open("saida.json", "w", encoding="utf-8") as f:
    json.dump(dados, f, indent=2)
```

Muito util para:

- salvar melhor rede
- salvar resultados de treino
- registrar metricas

## 4. `os` e caminhos de arquivo

Projetos reais precisam localizar arquivos com seguranca.

Exemplo:

```python
import os
base = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base, "config.json")
```

## 5. Por que isso importa

Se voce usar caminhos relativos ingênuos, o programa pode funcionar num lugar e quebrar em outro.

## 6. `sys` e ambiente de execucao

No projeto, `sys` ajuda inclusive com empacotamento via PyInstaller, usando logica como `_MEIPASS`.

Esse ponto e importante para entender que codigo de projeto real lida com mais do que o algoritmo puro.

## 7. Debug tecnico em Python

Debug aqui tem pelo menos 4 niveis:

### Nivel 1: logs simples

```python
print(carro.x, carro.y, carro.velocidade)
```

### Nivel 2: checagem de tipos e shapes

```python
print(sensores.shape)
print(W1.shape)
```

### Nivel 3: isolamento de funcoes

Testar separadamente:

- sensores
- colisao
- forward pass
- mutacao

### Nivel 4: visualizacao

Desenhar sensores e trajetoria para validar intuicao.

## 8. Assertions como ferramenta mental

```python
assert len(sensores) == 8
```

Serve para proteger suposicoes importantes.

## 9. O que salvar em arquivo quando estiver aprendendo

- melhor fitness por geracao
- media por geracao
- pesos da melhor rede
- configuracao usada no experimento

Isso torna aprendizado e comparacao muito mais solidos.

## 10. Dica de maturidade

Nao trate configuracao e debug como detalhes secundarios. Em projeto de IA, eles sao parte do motor de aprendizagem humana do desenvolvedor.

## Exercicios

### Exercicio 1

Crie um `config.json` seu com pelo menos 8 hiperparametros e carregue-os em um script Python.

### Exercicio 2

Implemente um pequeno logger em JSON que salve, a cada geracao, o melhor fitness e a media de fitness do seu experimento.
