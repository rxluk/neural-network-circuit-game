# Modulo Bibliotecas 03: tkinter, JSON e empacotamento

Este modulo cobre ferramentas auxiliares que aparecem no projeto e que ajudam a fechar o ciclo de uso real.

## 1. tkinter no projeto

Aqui, tkinter aparece apenas para suporte de clipboard no editor de pista.

Ou seja: nao e a UI principal do simulador. E um apoio pontual.

## 2. O que voce precisa saber de tkinter para este caso

- criar um `Tk()` temporario
- acessar clipboard
- destruir a janela auxiliar

Esse e um uso bem pequeno e pragmatitco da biblioteca.

## 3. JSON como contrato de dados

No projeto, JSON nao e so armazenamento. Ele e um contrato entre componentes.

Exemplos:

- `pista.json` define o ambiente
- `config.json` define hiperparametros
- `resultados.json` registra saida do treino

## 4. Empacotamento com PyInstaller

O arquivo `.spec` existe para transformar o projeto em executavel.

Esse e um ponto importante de maturidade de projeto: pensar distribuicao, nao apenas implementacao.

## 5. O que o `.spec` te ensina

- quais arquivos precisam ser embarcados
- quais dependencias sao realmente necessarias
- como excluir bibliotecas pesadas nao usadas

## 6. Detalhe interessante deste projeto

PyTorch esta nas dependencias, mas o build exclui `torch`, `torchvision` e `torchaudio` porque nao sao usados em runtime nessa versao.

Isso mostra consciencia de engenharia.

## Exercicios

### Exercicio 1

Crie um pequeno script que leia `config.json`, gere `resultado.json` e depois releia esse resultado para imprimir um resumo.

### Exercicio 2

Leia o arquivo `.spec` do projeto e escreva um mini-resumo explicando por que empacotamento exige pensar em recursos, dependencias e caminhos.
