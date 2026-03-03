"""
Simulation engine — no UI or rendering.
SimuladorBase handles the genetic-algorithm training loop.
"""

import numpy as np
import json
import os

from .track import (
    CONFIG, _COR, _CENTERLINE,
    CarrinhoIA, calcular_tangente_pista,
)
from .neural_network import RedeNeuralCarrinho


class SimuladorBase:
    """Genético + loop de simulação, sem qualquer dependência de matplotlib."""

    def __init__(self):
        self.populacao_size = CONFIG['simulacao']['populacao']
        self.carrinhos      = [CarrinhoIA(i, self.populacao_size)
                               for i in range(self.populacao_size)]
        self.cerebros       = [RedeNeuralCarrinho() for _ in range(self.populacao_size)]

        self.geracao                       = 0
        self.frame_atual                   = 0
        self.historico_melhor              = []
        self.historico_media               = []
        self.melhor_fitness_geral          = 0
        self.record_global_volta           = None
        self.record_global_volta_geracao   = None
        self.vencedor                      = None
        self.gens_sem_melhora              = 0
        self._melhor_fitness_ultimo        = 0
        self._record_cl_index              = 0
        self.taxa_mutacao_atual            = 0.5
        self.modo_mutacao                  = 'explore'
        self.avg_laps_gen                  = 0.0
        self.parar_animacao                = False

    # ------------------------------------------------------------------
    # Tentativa (uma geração em execução)
    # ------------------------------------------------------------------

    def iniciar_tentativa(self):
        """Reseta todos os carros para o início de uma nova tentativa."""
        for c in self.carrinhos:
            c.reset()
        self.frame_atual = 0

    # ------------------------------------------------------------------
    # Simulação frame a frame
    # ------------------------------------------------------------------

    def simular_frame(self) -> bool:
        """Avança um frame. Retorna True quando a tentativa termina."""
        cfg_sim   = CONFIG['simulacao']
        rec       = CONFIG['recompensas']
        pen_cfg   = CONFIG['penalidades']
        _cl_total = len(_CENTERLINE)
        vivos     = 0

        sobreviventes_idx  = []
        sobreviventes_sens = []

        for i, (carrinho, _) in enumerate(zip(self.carrinhos, self.cerebros)):
            if not carrinho.vivo:
                continue
            if carrinho.voltas_completas >= cfg_sim['voltas_objetivo']:
                self.vencedor = i
                return True

            vivos += 1
            sensores = carrinho.get_sensores()
            carrinho._cached_sensores  = sensores
            carrinho.estado_frame      = {}
            carrinho._pts_frame_inicio = carrinho.pontos_acumulados

            # Detecta contramão
            _rad_cw  = np.radians(carrinho.angulo)
            _vel_dir = np.array([np.cos(_rad_cw), np.sin(_rad_cw)])
            _tang    = calcular_tangente_pista(carrinho.x, carrinho.y)
            _dot     = float(np.dot(_vel_dir, _tang))
            if _dot < pen_cfg['contramao_limiar']:
                carrinho.estado_frame['contramao'] = True
                if carrinho.voltas_completas > 0:
                    _pen_cw = pen_cfg['penalidade_contramao'] * (-_dot)
                    carrinho.pontos_acumulados -= _pen_cw
                    if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wrong way!':
                        carrinho.eventos.append((f'-{_pen_cw:.2f} Wrong way!', _COR['vermelho']))
                else:
                    if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wrong way!':
                        carrinho.eventos.append(('Freeze Wrong way!', _COR['amarelo']))

            # Kill por estagnação
            carrinho.frames_checkpoint += 1
            janela = cfg_sim['frames_sem_progresso']
            if carrinho.frames_checkpoint >= janela:
                avanco_cl = carrinho.max_cl_index - carrinho.cl_checkpoint
                if avanco_cl < cfg_sim['cl_min_progresso_por_janela']:
                    carrinho.vivo         = False
                    carrinho.motivo_morte = 'No progress'
                    carrinho.eventos.append(('Stagnant', _COR['vermelho']))
                    continue
                carrinho.cl_checkpoint     = carrinho.max_cl_index
                carrinho.frames_checkpoint = 0

            # Recompensa por sensor (proximidade ao centro e velocidade)
            vel_norm   = carrinho.velocidade / CONFIG['carros']['velocidade_max']
            min_sensor = float(sensores[:7].min())
            carrinho.pontos_acumulados += min_sensor * vel_norm * rec['peso_centro']

            frontal_min = float(sensores[2:5].min())
            limiar      = CONFIG['pista']['largura_pista'] / 15.0
            if carrinho.voltas_completas > 0 and frontal_min < limiar:
                pen_val = pen_cfg['penalidade_parede_proxima'] * (limiar - frontal_min) / limiar
                carrinho.pontos_acumulados -= pen_val
                carrinho.estado_frame['parede'] = True
                if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wall!':
                    carrinho.eventos.append((f'-{pen_val:.2f} Wall!', _COR['vermelho']))

            sobreviventes_idx.append(i)
            sobreviventes_sens.append(sensores)

        # Batch forward pass
        if sobreviventes_idx:
            sensor_matrix = np.stack(sobreviventes_sens)
            redes_vivas   = [self.cerebros[i] for i in sobreviventes_idx]
            acoes_batch   = RedeNeuralCarrinho.forward_batch(redes_vivas, sensor_matrix)

            for k, i in enumerate(sobreviventes_idx):
                carrinho = self.carrinhos[i]
                acao     = acoes_batch[k]
                carrinho.ultima_acao = acao
                carrinho.mover(acao)

                # Atualiza record global de volta
                if carrinho.estado_frame.get('volta') and carrinho.melhor_tempo_volta is not None:
                    if (self.record_global_volta is None
                            or carrinho.melhor_tempo_volta < self.record_global_volta):
                        self.record_global_volta          = carrinho.melhor_tempo_volta
                        self.record_global_volta_geracao  = self.geracao + 1
                        CONFIG['recompensas']['target_frames_volta'] = self.record_global_volta

                # Progresso na centerline
                dists2  = (_CENTERLINE[:, 0] - carrinho.x)**2 + (_CENTERLINE[:, 1] - carrinho.y)**2
                cl_idx  = int(np.argmin(dists2))
                prev_cl = carrinho.max_cl_index % _cl_total
                diff_cl = (cl_idx - prev_cl) % _cl_total
                if 0 < diff_cl < _cl_total // 2:
                    vel_norm  = carrinho.velocidade / CONFIG['carros']['velocidade_max']
                    fator_vel = CONFIG['recompensas']['fator_velocidade_progresso']
                    vel_mult  = 1.0 - fator_vel + fator_vel * vel_norm
                    pts = diff_cl * rec['peso_progresso_circuito'] * vel_mult
                    carrinho.pontos_acumulados += pts
                    carrinho.max_cl_index      += diff_cl
                    if diff_cl >= 3:
                        carrinho.eventos.append((f'+{pts:.0f} Progress', _COR['verde']))

                carrinho.checar_colisao()

                # Contramão bloqueia ganhos do frame
                if carrinho.estado_frame.get('contramao'):
                    carrinho.pontos_acumulados = min(
                        carrinho.pontos_acumulados, carrinho._pts_frame_inicio)

        self.frame_atual += 1
        return vivos == 0

    # ------------------------------------------------------------------
    # Evolução genética
    # ------------------------------------------------------------------

    def evoluir_geracao(self):
        """Seleciona, cruza e muta para a próxima geração."""
        fitness = [c.pontos_acumulados for c in self.carrinhos]

        target = self.record_global_volta or CONFIG['recompensas']['target_frames_volta']
        for i, c in enumerate(self.carrinhos):
            if c.melhor_tempo_volta is not None and c.voltas_completas >= 1:
                speed_factor = target / c.melhor_tempo_volta
                fitness[i]  *= max(0.4, min(2.5, speed_factor))

        melhor_fitness = max(fitness)
        media_fitness  = np.mean(fitness)

        self.historico_melhor.append(melhor_fitness)
        self.historico_media.append(media_fitness)

        if melhor_fitness > self.melhor_fitness_geral:
            self.melhor_fitness_geral = melhor_fitness

        melhor_cl_gen = max(c.max_cl_index for c in self.carrinhos)
        cl_avancou    = melhor_cl_gen > self._record_cl_index
        if cl_avancou:
            self._record_cl_index = melhor_cl_gen

        if melhor_fitness > self._melhor_fitness_ultimo * 1.005 or cl_avancou:
            self.gens_sem_melhora          = 0
            self._melhor_fitness_ultimo    = melhor_fitness
        else:
            self.gens_sem_melhora += 1

        self.avg_laps_gen = float(np.mean([c.voltas_completas for c in self.carrinhos]))

        tempos = [c.melhor_tempo_volta for c in self.carrinhos if c.melhor_tempo_volta is not None]
        if tempos:
            melhor_gen = min(tempos)
            if self.record_global_volta is None or melhor_gen < self.record_global_volta:
                self.record_global_volta         = melhor_gen
                self.record_global_volta_geracao = self.geracao + 1

        cfg   = CONFIG['simulacao']
        top_n = cfg['top_sobreviventes']

        indices_ordenados = np.argsort(fitness)[::-1]

        stag          = self.gens_sem_melhora
        plato_critico = stag >= 60
        plato_grave   = stag >= 30
        plato_leve    = stag >= 15

        if plato_critico:
            taxa_mutacao = 0.50
            n_aleatorios = 15
            top_n        = 8
            modo         = 'CRITICAL'
        elif plato_grave:
            taxa_mutacao = 0.38
            n_aleatorios = min(12, self.populacao_size // 4)
            top_n        = 6
            modo         = 'plateau'
        elif plato_leve:
            taxa_mutacao = 0.28
            n_aleatorios = cfg['novos_aleatorios_por_geracao']
            modo         = 'shake'
        elif self.geracao < 10:
            taxa_mutacao = 0.40
            n_aleatorios = cfg['novos_aleatorios_por_geracao']
            modo         = 'explore'
        elif self.geracao < 30:
            taxa_mutacao = 0.28
            n_aleatorios = cfg['novos_aleatorios_por_geracao']
            modo         = 'normal'
        else:
            taxa_mutacao = 0.18
            n_aleatorios = cfg['novos_aleatorios_por_geracao']
            modo         = 'fine'

        self.taxa_mutacao_atual = taxa_mutacao
        self.modo_mutacao       = modo

        melhores      = indices_ordenados[:top_n]
        novos_cerebros = []

        # Elitismo
        for idx in melhores:
            novo = RedeNeuralCarrinho()
            novo.copiar_de(self.cerebros[idx])
            novos_cerebros.append(novo)

        # Novos aleatórios
        for _ in range(n_aleatorios):
            novos_cerebros.append(RedeNeuralCarrinho())

        # Filhos: crossover + mutação
        n_pais = max(2, len(melhores))
        while len(novos_cerebros) < self.populacao_size:
            pai_idx = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
            filho   = RedeNeuralCarrinho()
            filho.copiar_de(self.cerebros[pai_idx])
            if np.random.rand() < 0.5 and len(melhores) >= 2:
                outro_idx = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
                while outro_idx == pai_idx and len(melhores) > 1:
                    outro_idx = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
                filho.crossover(self.cerebros[outro_idx])
            filho.mutar(taxa_mutacao)
            novos_cerebros.append(filho)

        self.cerebros = novos_cerebros
        self.geracao += 1

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _salvar_resultados(self):
        """Salva estatísticas e pesos da rede vencedora em resultados.json."""
        from datetime import datetime
        venc  = self.carrinhos[self.vencedor]
        rede  = self.cerebros[self.vencedor]
        dados = {
            'data':                        datetime.now().isoformat(timespec='seconds'),
            'geracao':                     self.geracao + 1,
            'voltas_objetivo':             CONFIG['simulacao']['voltas_objetivo'],
            'record_melhor_volta_frames':  self.record_global_volta,
            'carro_vencedor':              self.vencedor + 1,
            'pontos_vencedor':             float(venc.pontos_acumulados),
            'rede_neural': {
                'W1': rede.W1.tolist(),
                'b1': rede.b1.tolist(),
                'W2': rede.W2.tolist(),
                'b2': rede.b2.tolist(),
            },
            'simulacao': {
                'populacao':  CONFIG['simulacao']['populacao'],
                'top_sobrev': CONFIG['simulacao']['top_sobreviventes'],
            },
        }
        caminho = os.path.join(os.path.dirname(__file__), '..', 'resultados.json')
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2)
        print(f'[resultados] Salvo em {caminho}')
        return caminho, dados
