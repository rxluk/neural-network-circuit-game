# Gabarito 04: Algoritmos geneticos e reward

1. Fitness e a medida numerica da qualidade do comportamento do agente.
2. Elitismo e copiar os melhores diretamente para a proxima geracao, preservando boas solucoes.
3. Crossover uniforme escolhe, para cada gene ou parametro, se ele vem do pai A ou do pai B.
4. Mutacao com ruido gaussiano soma pequenas perturbacoes aleatorias aos parametros.
5. Exploracao e testar possibilidades novas; explotacao e refinar o que ja funciona.
6. Uma reward minima boa para corredor inclui: progresso positivo, penalidade forte por colisao e bonus leve por velocidade util.
7. Reward hacking no corredor: o agente pode aprender a ficar tremendo ou explorando uma brecha local em vez de progredir. Mitigacao: premiar progresso real e nao apenas tempo vivo.
8. Um pseudocodigo bom precisa conter: avaliar todos, ordenar por fitness, manter elite, gerar filhos por crossover, mutar filhos, formar nova populacao, repetir.
