# Motor de simulação sem nenhuma dependência de UI.
# SimuladorBase gerencia o loop de treinamento com algoritmo genético.

import numpy as np
import json
import os

from .track import (
    CONFIG, _COR, _CENTERLINE,
    CarrinhoIA, calcular_tangente_pista,
)
from .neural_network import RedeNeuralCarrinho


class SimuladorBase:

    def __init__(self):
        self.populacao_size = CONFIG['simulacao']['populacao']
        self.carrinhos = [CarrinhoIA(i, self.populacao_size)
                       for i in range(self.populacao_size)]
        self.cerebros = [RedeNeuralCarrinho() for _ in range(self.populacao_size)]

        self.geracao = 0
        self.frame_atual = 0
        self.historico_melhor = []
        self.historico_media = []
        self.melhor_fitness_geral = 0
        self.record_global_volta = None
        self.record_global_volta_geracao = None
        self.vencedor = None
        self.gens_sem_melhora = 0
        self._melhor_fitness_ultimo = 0
        self._record_cl_index = 0
        self.taxa_mutacao_atual = 0.5
        self.modo_mutacao = 'explore'
        self.avg_laps_gen = 0.0
        self.parar_animacao = False

    def iniciar_tentativa(self):
        for c in self.carrinhos:
            c.reset()
        self.frame_atual = 0

    def simular_frame(self) -> bool:
        """Avança um frame. Retorna True quando a tentativa termina."""
        cfg_sim = CONFIG['simulacao']
        rec = CONFIG['recompensas']
        pen_cfg = CONFIG['penalidades']
        cl_total = len(_CENTERLINE)
        vivos = 0

        sobreviventes_idx = []
        sobreviventes_sens = []

        for i, carrinho in enumerate(self.carrinhos):
            if not carrinho.vivo:
                continue
            if carrinho.voltas_completas >= cfg_sim['voltas_objetivo']:
                self.vencedor = i
                return True

            vivos += 1
            sensores = carrinho.get_sensores()
            carrinho._cached_sensores = sensores
            carrinho.estado_frame = {}
            carrinho._pts_frame_inicio = carrinho.pontos_acumulados

            self._checar_contramao(carrinho, pen_cfg)

            if self._checar_estagnacao(carrinho, cfg_sim):
                continue

            self._recompensa_sensores(carrinho, sensores, rec, pen_cfg)
            sobreviventes_idx.append(i)
            sobreviventes_sens.append(sensores)

        # Forward pass vetorizado para todos os sobreviventes ao mesmo tempo
        if sobreviventes_idx:
            sensor_matrix = np.stack(sobreviventes_sens)
            redes_vivas = [self.cerebros[i] for i in sobreviventes_idx]
            acoes_batch = RedeNeuralCarrinho.forward_batch(redes_vivas, sensor_matrix)

            for k, i in enumerate(sobreviventes_idx):
                carrinho = self.carrinhos[i]
                carrinho.ultima_acao = acoes_batch[k]
                carrinho.mover(acoes_batch[k])

                self._atualizar_record_volta(carrinho)
                self._atualizar_progresso_cl(carrinho, rec, cl_total)
                carrinho.checar_colisao()

                # Contramão anula qualquer ganho de pontos no frame
                if carrinho.estado_frame.get('contramao'):
                    carrinho.pontos_acumulados = min(
                        carrinho.pontos_acumulados, carrinho._pts_frame_inicio)

        self.frame_atual += 1
        return vivos == 0

    def _checar_contramao(self, carrinho, pen_cfg):
        """Detecta se o carro está andando na direção errada e aplica penalidade."""
        rad = np.radians(carrinho.angulo)
        vel_dir = np.array([np.cos(rad), np.sin(rad)])
        tang = calcular_tangente_pista(carrinho.x, carrinho.y)
        dot = float(np.dot(vel_dir, tang))

        if dot < pen_cfg['contramao_limiar']:
            carrinho.estado_frame['contramao'] = True
            if carrinho.voltas_completas > 0:
                pen = pen_cfg['penalidade_contramao'] * (-dot)
                carrinho.pontos_acumulados -= pen
                if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wrong way!':
                    carrinho.eventos.append((f'-{pen:.2f} Wrong way!', _COR['vermelho']))
            else:
                if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wrong way!':
                    carrinho.eventos.append(('Freeze Wrong way!', _COR['amarelo']))

    def _checar_estagnacao(self, carrinho, cfg_sim) -> bool:
        """Mata o carro se não houve progresso na centerline. Retorna True se eliminou."""
        carrinho.frames_checkpoint += 1
        if carrinho.frames_checkpoint < cfg_sim['frames_sem_progresso']:
            return False

        avanco = carrinho.max_cl_index - carrinho.cl_checkpoint
        if avanco < cfg_sim['cl_min_progresso_por_janela']:
            carrinho.vivo = False
            carrinho.motivo_morte = 'No progress'
            carrinho.eventos.append(('Stagnant', _COR['vermelho']))
            return True

        carrinho.cl_checkpoint = carrinho.max_cl_index
        carrinho.frames_checkpoint = 0
        return False

    def _recompensa_sensores(self, carrinho, sensores, rec, pen_cfg):
        """Recompensa por estar no centro e penaliza por parede muito próxima."""
        vel_norm = carrinho.velocidade / CONFIG['carros']['velocidade_max']
        min_sensor = float(sensores[:7].min())
        carrinho.pontos_acumulados += min_sensor * vel_norm * rec['peso_centro']

        frontal_min = float(sensores[2:5].min())
        limiar = CONFIG['pista']['largura_pista'] / 15.0
        if carrinho.voltas_completas > 0 and frontal_min < limiar:
            pen_val = pen_cfg['penalidade_parede_proxima'] * (limiar - frontal_min) / limiar
            carrinho.pontos_acumulados -= pen_val
            carrinho.estado_frame['parede'] = True
            if not carrinho.eventos or carrinho.eventos[-1][0] != 'Wall!':
                carrinho.eventos.append((f'-{pen_val:.2f} Wall!', _COR['vermelho']))

    def _atualizar_record_volta(self, carrinho):
        if carrinho.estado_frame.get('volta') and carrinho.melhor_tempo_volta is not None:
            if (self.record_global_volta is None
                    or carrinho.melhor_tempo_volta < self.record_global_volta):
                self.record_global_volta = carrinho.melhor_tempo_volta
                self.record_global_volta_geracao = self.geracao + 1
                CONFIG['recompensas']['target_frames_volta'] = self.record_global_volta

    def _atualizar_progresso_cl(self, carrinho, rec, cl_total):
        """Adiciona pontos conforme o carro avança na centerline."""
        dists2 = (_CENTERLINE[:, 0] - carrinho.x)**2 + (_CENTERLINE[:, 1] - carrinho.y)**2
        cl_idx = int(np.argmin(dists2))
        prev_cl = carrinho.max_cl_index % cl_total
        diff_cl = (cl_idx - prev_cl) % cl_total

        if 0 < diff_cl < cl_total // 2:
            vel_norm = carrinho.velocidade / CONFIG['carros']['velocidade_max']
            fv = rec['fator_velocidade_progresso']
            vel_mult = 1.0 - fv + fv * vel_norm
            pts = diff_cl * rec['peso_progresso_circuito'] * vel_mult
            carrinho.pontos_acumulados += pts
            carrinho.max_cl_index      += diff_cl
            if diff_cl >= 3:
                carrinho.eventos.append((f'+{pts:.0f} Progress', _COR['verde']))

    def evoluir_geracao(self):
        """Seleciona, cruza e muta para a próxima geração."""
        fitness = self._calcular_fitness()
        melhor  = max(fitness)
        media   = float(np.mean(fitness))

        self.historico_melhor.append(melhor)
        self.historico_media.append(media)
        if melhor > self.melhor_fitness_geral:
            self.melhor_fitness_geral = melhor

        melhor_cl = max(c.max_cl_index for c in self.carrinhos)
        cl_avancou = melhor_cl > self._record_cl_index
        if cl_avancou:
            self._record_cl_index = melhor_cl

        if melhor > self._melhor_fitness_ultimo * 1.005 or cl_avancou:
            self.gens_sem_melhora = 0
            self._melhor_fitness_ultimo = melhor
        else:
            self.gens_sem_melhora += 1

        self.avg_laps_gen = float(np.mean([c.voltas_completas for c in self.carrinhos]))

        tempos = [c.melhor_tempo_volta for c in self.carrinhos if c.melhor_tempo_volta is not None]
        if tempos:
            melhor_gen = min(tempos)
            if self.record_global_volta is None or melhor_gen < self.record_global_volta:
                self.record_global_volta = melhor_gen
                self.record_global_volta_geracao = self.geracao + 1

        taxa, n_rand, top_n, modo = self._params_mutacao()
        self.taxa_mutacao_atual = taxa
        self.modo_mutacao = modo

        melhores = np.argsort(fitness)[::-1][:top_n]
        self.cerebros = self._construir_nova_geracao(melhores, taxa, n_rand)
        self.geracao += 1

    def _calcular_fitness(self):
        """Ajusta o fitness bruto pelo tempo de volta para premiar velocidade."""
        fitness = [c.pontos_acumulados for c in self.carrinhos]
        target = self.record_global_volta or CONFIG['recompensas']['target_frames_volta']
        for i, c in enumerate(self.carrinhos):
            if c.melhor_tempo_volta is not None and c.voltas_completas >= 1:
                fator = target / c.melhor_tempo_volta
                fitness[i] *= max(0.4, min(2.5, fator))
        return fitness

    def _params_mutacao(self):
        """Retorna (taxa, n_aleatorios, top_n, modo) de acordo com a estagnação atual."""
        stag = self.gens_sem_melhora
        cfg = CONFIG['simulacao']
        top_n = cfg['top_sobreviventes']
        n_pad = cfg['novos_aleatorios_por_geracao']

        if stag >= 60:  return 0.50, 15, 8, 'CRITICAL'
        if stag >= 30:  return 0.38, min(12, self.populacao_size // 4), 6, 'plateau'
        if stag >= 15:  return 0.28, n_pad, top_n, 'shake'
        if self.geracao < 10: return 0.40, n_pad, top_n, 'explore'
        if self.geracao < 30: return 0.28, n_pad, top_n, 'normal'
        return 0.18, n_pad, top_n, 'fine'

    def _construir_nova_geracao(self, melhores, taxa, n_rand):
        """Elitismo + novos aleatórios + filhos por crossover e mutação."""
        novos = []
        n_pais = max(2, len(melhores))

        for idx in melhores:
            copia = RedeNeuralCarrinho()
            copia.copiar_de(self.cerebros[idx])
            novos.append(copia)

        for _ in range(n_rand):
            novos.append(RedeNeuralCarrinho())

        while len(novos) < self.populacao_size:
            pai = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
            filho = RedeNeuralCarrinho()
            filho.copiar_de(self.cerebros[pai])

            if np.random.rand() < 0.5 and len(melhores) >= 2:
                outro = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
                while outro == pai and len(melhores) > 1:
                    outro = melhores[np.random.randint(0, min(n_pais, len(melhores)))]
                filho.crossover(self.cerebros[outro])

            filho.mutar(taxa)
            novos.append(filho)

        return novos

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
