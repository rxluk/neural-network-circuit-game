# Gabarito 01: Python e arquitetura

1. Estado e o que o objeto guarda; comportamento e o que ele faz. Em `Carro`, `x`, `y`, `angulo` e `velocidade` sao estado. `mover()` e `get_sensores()` sao comportamento.
2. A classe `Carro` deve encapsular posicao e movimento. Uma resposta boa usa `self.x`, `self.y`, `self.angulo` e `self.velocidade`.
3. A classe `Rede` deve guardar pesos e bias e copiar arrays com `.copy()` para evitar referencia compartilhada.
4. Uma estrutura boa separa entrada, configuracao e modulos de dominio. Exemplo: `main.py`, `config.json`, `sim/track.py`, `sim/network.py`, `sim/evolution.py`, `sim/render.py`.
5. Uma resposta boa usa `json.load()` e acessa ao menos 3 chaves do dicionario.
6. Separar motor e visualizacao melhora teste, clareza, reuso e permite rodar treino sem interface.
