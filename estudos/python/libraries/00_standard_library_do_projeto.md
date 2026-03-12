# Modulo Bibliotecas 00: Standard library usada no projeto

Muita gente subestima a standard library do Python. Mas projeto bom costuma depender muito dela.

## 1. `json`

Serve para ler e escrever configuracoes e resultados.

No projeto, ele aparece para:

- carregar `config.json`
- carregar `pista.json`
- salvar resultados do treino

### Metodos principais

- `json.load(f)`
- `json.dump(obj, f, indent=2)`
- `json.loads(texto)`
- `json.dumps(obj)`

## 2. `os`

Serve para lidar com caminhos, diretorios e ambiente do sistema operacional.

### Funcoes muito importantes aqui

- `os.path.join(...)`
- `os.path.dirname(...)`
- `os.path.abspath(...)`
- `os.path.exists(...)`

Sem isso, localizar arquivos do projeto de forma robusta fica ruim.

## 3. `sys`

Serve para acessar detalhes do ambiente Python atual.

No projeto, e importante para compatibilidade com PyInstaller.

### Exemplo de uso conceitual

- detectar se o programa esta empacotado
- resolver caminho base de recursos

## 4. `collections.deque`

`deque` e muito util quando voce quer manter uma janela deslizante eficiente.

No projeto, isso aparece para guardar historico curto de pontuacao e detectar perda continua.

### Por que usar `deque` em vez de lista comum?

Porque inserir/remover nas pontas e eficiente e o `maxlen` torna o buffer automatico.

## 5. `datetime`

Usado para registrar quando o resultado foi salvo.

## 6. `math` vs `numpy`

Em projeto numerico, `numpy` costuma dominar, mas `math` ainda pode ser util para funcoes escalares simples.

## 7. `random` vs `numpy.random`

Em projetos vetoriais, prefira `numpy.random` quando o dado principal ja esta em arrays.

## 8. Conclusao pratica

A standard library nao e acessorio. Ela e a cola que conecta o sistema todo.

## Exercicios

### Exercicio 1

Implemente um script que carregue um JSON de configuracao, altere um valor e salve outro JSON com timestamp.

### Exercicio 2

Crie uma janela deslizante com `deque(maxlen=5)` e use-a para acompanhar os ultimos 5 scores de um agente.
