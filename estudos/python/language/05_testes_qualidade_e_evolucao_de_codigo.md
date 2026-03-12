# Modulo Python 05: Testes, qualidade e evolucao de codigo

Quanto mais voce evoluir seu proprio projeto, mais vai sentir necessidade de controlar qualidade.

## 1. Por que qualidade importa aqui

Projetos de IA e simulacao acumulam bugs sorrateiros:

- shape errado
- escala errada
- reward incoerente
- mutacao afetando referencia compartilhada

## 2. Tipos de verificacao uteis

- asserts
- testes unitarios pequenos
- logs de sanidade
- visualizacao de inspecao

## 3. O que testar primeiro

- sensores retornam tamanho esperado
- forward retorna shapes esperados
- colisao funciona em casos obvios
- mutacao realmente altera pesos
- copia de rede nao compartilha memoria por acidente

## 4. Refatoracao consciente

Refatorar nao e mudar tudo. E melhorar estrutura preservando comportamento desejado.

## 5. Sinal de maturidade

Quando voce para de depender apenas de "rodei e pareceu certo" e comeca a verificar componentes isolados.

## Exercicios

### Exercicio 1

Liste 5 invariantes que deveriam ser verdadeiras no seu proprio simulador.

### Exercicio 2

Escreva 3 testes pequenos que validem sensores, forward pass e mutacao.
