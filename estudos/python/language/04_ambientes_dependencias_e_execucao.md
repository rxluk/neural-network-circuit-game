# Modulo Python 04: Ambientes, dependencias e execucao

Saber programar de verdade nao e so escrever arquivos `.py`. Tambem e saber como rodar o projeto de forma repetivel.

## 1. O que e ambiente Python

Ambiente e o conjunto de:

- interpretador Python
- pacotes instalados
- versoes das bibliotecas
- caminhos e contexto de execucao

## 2. Por que ambientes importam

Porque o mesmo codigo pode funcionar em uma maquina e quebrar em outra se versoes mudarem.

## 3. `venv`

O mecanismo padrao simples do Python para isolar dependencias.

Fluxo tipico:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. `requirements.txt`

Ele registra dependencias do projeto.

Neste projeto, ele mostra o ecossistema esperado para rodar a simulacao.

## 5. Scripts de entrada

Sempre saiba qual arquivo inicia o sistema e quais sao os scripts auxiliares.

No projeto atual:

- `rede_neural_jogo.py` inicia a simulacao
- `editor_pista.py` inicia o editor de pista

## 6. Versao do Python

Projetos numericos e visuais podem depender de versoes especificas. Por isso o README explicita Python 3.11+.

## 7. Dependencias opcionais vs necessarias

Nem tudo que aparece em dependencias e central ao runtime do projeto.

Aqui, NumPy e Matplotlib sao centrais. PyTorch aparece como opcional/futuro.

## 8. Reprodutibilidade

Saber programar para ML inclui se preocupar com:

- versao de Python
- versao de bibliotecas
- seed aleatoria
- configuracao do experimento

## Exercicios

### Exercicio 1

Crie um `venv` para um mini-projeto seu, instale `numpy` e execute um script dentro dele.

### Exercicio 2

Escreva um mini-checklist de reproducao para rodar este projeto em outra maquina.
