# Trilha: Fundações Matemáticas para IA/ML

## Propósito

Este bloco estabelece **fundações absolutamente rigorosas** para aprender redes neurais no nível de mestrado.

Sem essas fundações, você vai:
- Não entender backpropagation (mágica)
- Errar em implementações de gradientes
- Não saber debugar problemas numéricos
- Não conseguir ler papers de ML

## Estrutura

1. **00 - Introdução e Filosofia** 
   - Por que matemática importa
   - Roadmap de aprendizado
   - Convenções de notação

2. **01 - Álgebra Linear Profunda**
   - Vetores, normas, produto interno
   - Matrizes, transformações, operações
   - Determinante, inversa, sistemas lineares
   - Autovalores/autovetores
   - Decomposições (Eigen, SVD)
   - **Prática:** Cálculos em papel + NumPy

3. **02 - Cálculo Vetorial para ML**
   - Derivadas parciais
   - Gradientes (vetor de derivadas)
   - Matriz Jacobiana
   - Regra da cadeia (ESSENCIAL)
   - Hessiana (segunda derivada)
   - Verificação numérica de derivadas
   - **Aplicação:** Gradient descent, interpretação geométrica

4. **03 - Probabilidade e Entropia** *(em progresso)*
   - Distribuições (normal, bernoulli, categórica)
   - Entropia, cross-entropy
   - Máxima verossimilhança
   - KL divergence

5. **04 - Análise Numérica** *(em progresso)*
   - Estabilidade de algoritmos
   - Overflow/underflow
   - Log-sum-exp trick
   - Floating-point precision

## Como Usar

### Cenário 1: Você conhece conceitos mas quer rigor

- Leia seções de "Exemplo Concreto"
- Pule algumas derivações se confiante
- Foque em "Utilização em ML"

###Cenário 2: Você é novo em matemática

- Comece do início, com papel e caneta
- Faça todos os exercícios, validate com código
- Permita 2-3 horas por módulo

### Cenário 3: Você quer speed-run

- Enfoque "Interpretação Geométrica"
- Leia "Exemplo Concreto"
- Rode os notebooks
- Resolva exercícios críticos

## Checklist de Domínio

- [ ] Diferencio norma L1 de L2 e quando usar cada uma?
- [ ] Consigo calcular à mão produto de matrizes?
- [ ] Entendo o que é determinante geometricamente?
- [ ] Sei o que é autovetor intuitivamente?
- [ ] Consigo derivar em papel sem erros?
- [ ] Entendo regra da cadeia profundamente?
- [ ] Posso implementar gradient descent do zero?
- [ ] Sei verificar derivada numericamente?
- [ ] Entendo matriz Jacobiana e shape?
- [ ] Consigo ler equações de papers sem pânico?

## Conexões Forward

Después completar este bloco, você vai para:

→ **Redes Neurais Profunda**
- Entender forward pass em termos matriciais
- Implementar backprop com **completa clareza**
- Debugar problemas de gradientes
- Entender otimizadores (SGD, Adam) matematicamente

→ **Probabilidade & Estatística**
- Entender loss functions (cross-entropy, KL divergence)
- Inferência Bayesiana
- Modelos probabilísticos

## Próximo Módulo

[→ 00 - Introdução e Filosofia](./00_intro_e_filosofia.md)
