# Matemática para IA/ML: Introdução, Filosofia e Mapa

## Por que matemática importa EM SÉRIO

Não é purismo acadêmico. É pragmático:

- **Backpropagation sem derivadas = mágica negra**, você não sabe por que funciona, erra nas implementações.
- **Álgebra linear sem conceitos = brincadeira**, você descobre iterativamente que `(2, 5) + (1, 5)` quebrarn (broadcasting confuso).
- **Probabilidade ausente = erros de design**, classificadores desbalanceados, métricas erradas.
- **Análise numérica ignorada = Bugs feios**, `exp(1000)` vira `inf`, modelo para de aprender.

Este bloco existe para que você:

1. **Entenda profundamente** por que operações em ML funcionam
2. **Conheça limites e armadilhas** numericamente
3. **Saiba fazer cálculos manualmente** (em papel) antes de código
4. **Conecte teoria ↔ prática** sem se perder

## Estrutura deste bloco matemático

```
├─ 00_intro_e_filosofia (você está aqui)
├─ 01_algebra_linear_profunda
│    ├ Vetores e normas
│    ├ Matrizes e transformações
│    ├ Valores/vetores próprios
│    ├ Decomposições (SVD, QR, Cholesky)
│    └ Aplicações em ML
├─ 02_calculo_vetorial_para_ml
│    ├ Derivadas parciais
│    ├ Gradientes e direção de máxima ascensão
│    ├ Matrix calculus rules
│    ├ Hessiano
│    └ Chain rule (regra da cadeia)
├─ 03_probabilidade_entropia
│    ├ Distribuições clássicas
│    ├ Entropia e informação
│    ├ Máxima verossimilhança
│    └ KL divergence
└─ 04_analise_numerica
     ├ Estabilidade de algoritmos
     ├ Overflow/underflow
     ├ Log-sum-exp trick
     └ Precisão numérica
```

## Diferença entre "matemática escolar" e "matemática para ML"

### Escola
- Resolve `2x + 5 = 0` → `x = -2.5`
- Calcula determinante à mão
- Memoriza fórmulas

### ML/Mestrado
- A variável `x` é **matriz**
- O "2" é **operador linear**
- A resposta é **interpretada geometricamente**
- Você **implementa numericamente** robusta contra arredondamento
- Sabe **quando usar truques** (log-sum-exp) para estabilidade

## Convenção de notação

Para evitar confusão, usamos:

| Símbolo | Significado | Exemplo |
|---------|-----------|---------|
| $x$ | escalar | $x = 5.0$ |
| $\mathbf{v}$ | vetor (coluna por padrão) | $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ |
| $\mathbf{M}$ | matriz | $\mathbf{M} \in \mathbb{R}^{m \times n}$ |
| $\mathbf{M}^T$ | transposta | troca linhas/colunas |
| $\mathbf{M}^{-1}$ | inversa (se existe) | $\mathbf{M} \mathbf{M}^{-1} = \mathbf{I}$ |
| $\mathbf{I}$ | matriz identidade | diagonal com 1s |
| $\\|\\mathbf{v}\\|$ | norma | comprimento |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ | produto interno | medida de similaridade |
| $\nabla f$ | gradiente | direção de máxima ascensão |
| $\nabla^2 f$ | Hessiano | curvatura |

## Roadmap de aprendizado

### Ordem recomendada (Cascata)

```
Álgebra Linear
    ↓
Cálculo Vetorial
    ↓
Backpropagation (derivadas em árvore)
    ↓
Otimizadores (SGD, Adam)
    ↓
Treinamento robusto
```

**Não tente pular etapas.** Entender álgebra linear é pré-requisito absoluto para cálculo vetorial, que é pré-requisito para backprop.

## Como usar este bloco

1. **Leia cada módulo com papel e caneta ao lado**
2. **Faça os cálculos manualmente** nos exercícios propostos
3. **Depois rode o código NumPy** para validar
4. **Conecte com próximo bloco** (NN trilha)

## Objetivo final deste bloco

Ao terminar, você será capaz de:

- ✅ Derivar fórmulas de backprop do zero
- ✅ Entender por que `(batch, features) @ (features, hidden)` dá `(batch, hidden)`
- ✅ Implementar SGD com momentum manualmente em NumPy
- ✅ Debugar problemas numéricos (gradiente muito pequeno?)
- ✅ Ler papers de ML sem se perder em notação

## Próximo módulo

[→ Álgebra Linear Profunda](./01_algebra_linear_profunda.md)
