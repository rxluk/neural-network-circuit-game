"""
Rede Neural Aprendendo a Dirigir - VISUALIZAÇÃO EM TEMPO REAL
Veja o carrinho tentando, batendo, e aprendendo com os erros!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import json
import os

# Carrega configurações do arquivo
def carregar_config():
    """Carrega configurações do arquivo config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = carregar_config()

def ponto_dentro_estadio(x, y, cx, cy, metade_reta, raio):
    """Verifica se um ponto (x,y) está dentro de uma pista estádio (reta + curvas semicirculares)"""
    # Projeta x no segmento central da reta, calcula distância ao eixo mais próximo
    px = max(cx - metade_reta, min(cx + metade_reta, x))
    return (x - px)**2 + (y - cy)**2 <= raio**2

def gerar_contorno_estadio(cx, cy, metade_reta, raio, n=120):
    """Gera os pontos do contorno de uma pista estádio"""
    # Curva direita: de -90 a +90 graus
    theta_dir = np.linspace(-np.pi/2, np.pi/2, n//2)
    # Curva esquerda: de +90 a +270 graus
    theta_esq = np.linspace(np.pi/2, 3*np.pi/2, n//2)
    xs = np.concatenate([
        cx + metade_reta + raio * np.cos(theta_dir),
        cx - metade_reta + raio * np.cos(theta_esq)
    ])
    ys = np.concatenate([
        cy + raio * np.sin(theta_dir),
        cy + raio * np.sin(theta_esq)
    ])
    return np.append(xs, xs[0]), np.append(ys, ys[0])

def calcular_tangente_pista(x, y, pista_config):
    """
    Retorna o vetor tangente na direcao CORRETA da pista no ponto (x, y).
    Algoritmo: projeta x sobre o eixo central da reta, calcula vetor radial
    para fora do eixo, rotaciona 90 graus no sentido anti-horario.
    """
    cx  = pista_config['centro_x']
    cy  = pista_config['centro_y']
    ml  = pista_config['metade_reta']
    # Ponto mais proximo no eixo horizontal da pista
    px  = max(cx - ml, min(cx + ml, x))
    rx  = x - px
    ry  = y - cy
    mag = np.sqrt(rx * rx + ry * ry)
    if mag < 1e-6:
        return np.array([1.0, 0.0])  # fallback
    rx /= mag
    ry /= mag
    # Rotacao 90 graus CCW: (rx, ry) -> (-ry, rx)
    return np.array([-ry, rx])


def calcular_posicao_inicial_carro(indice, total_carros):
    """Calcula posição inicial do carro dentro da linha de largada"""
    linha = CONFIG['linha_largada']
    
    # Centro da linha de largada
    x_centro = linha['x']
    y_centro = linha['y']
    
    # Largura utilizável da linha (percentual da largura total para não ficar nas bordas)
    largura_util = linha['largura'] * 0.8  # Usa 80% da largura para distribuir os carros
    
    # Distribui os carros ao longo da linha
    if total_carros == 1:
        offset_relativo = 0
    else:
        offset_relativo = (indice / (total_carros - 1) - 0.5)
    
    # Posição sem rotação (ao longo do eixo x da linha)
    x_local = offset_relativo * largura_util
    y_local = 0
    
    # Aplica rotação
    rotacao_rad = np.radians(linha['angulo'])
    x_rotacionado = x_local * np.cos(rotacao_rad) - y_local * np.sin(rotacao_rad)
    y_rotacionado = x_local * np.sin(rotacao_rad) + y_local * np.cos(rotacao_rad)
    
    # Posição final
    x_final = x_centro + x_rotacionado
    y_final = y_centro + y_rotacionado
    
    # Ângulo dos carros (direção inicial)
    angulo_inicial = CONFIG['carros']['angulo_inicial']
    
    return x_final, y_final, angulo_inicial

def esta_sobre_linha_largada(x, y):
    """Verifica se uma posição está sobre a linha de largada/chegada"""
    linha = CONFIG['linha_largada']
    
    # Centro da linha
    x_centro = linha['x']
    y_centro = linha['y']
    
    # Rotaciona o ponto para o sistema de coordenadas da linha (não rotacionada)
    rotacao_rad = -np.radians(linha['angulo'])  # Rotação inversa
    x_rel = x - x_centro
    y_rel = y - y_centro
    
    x_rot = x_rel * np.cos(rotacao_rad) - y_rel * np.sin(rotacao_rad)
    y_rot = x_rel * np.sin(rotacao_rad) + y_rel * np.cos(rotacao_rad)
    
    # Verifica se está dentro da linha (no sistema rotacionado)
    largura = linha['largura']
    altura = linha['altura']
    
    return bool((abs(x_rot) <= largura / 2) and (abs(y_rot) <= altura / 2))

def esta_sobre_linha_chegada(x, y):
    """Verifica se uma posição está sobre a linha de chegada"""
    linha = CONFIG['linha_chegada']
    
    # Centro da linha
    x_centro = linha['x']
    y_centro = linha['y']
    
    # Rotaciona o ponto para o sistema de coordenadas da linha (não rotacionada)
    rotacao_rad = -np.radians(linha['angulo'])  # Rotação inversa
    x_rel = x - x_centro
    y_rel = y - y_centro
    
    x_rot = x_rel * np.cos(rotacao_rad) - y_rel * np.sin(rotacao_rad)
    y_rot = x_rel * np.sin(rotacao_rad) + y_rel * np.cos(rotacao_rad)
    
    # Verifica se está dentro da linha (no sistema rotacionado)
    largura = linha['largura']
    altura = linha['altura']
    
    return bool((abs(x_rot) <= largura / 2) and (abs(y_rot) <= altura / 2))

class CarrinhoIA:
    """Carrinho controlado por rede neural"""
    
    def __init__(self, indice=0, total_carros=1):
        # Posição inicial baseada na linha de largada
        x, y, angulo = calcular_posicao_inicial_carro(indice, total_carros)
        self.x = x
        self.y = y
        self.velocidade = 0.0
        self.angulo = angulo
        self.vivo = True
        self.distancia_percorrida = 0.0
        self.tempo_vivo = 0
        self.trajetoria_x = []
        self.trajetoria_y = []
        self.voltas_completas = 0
        self.distancia_desde_ultima_volta = 0.0
        self.frame_inicio_volta = 0
        self.melhor_tempo_volta = None   # frames da volta mais rápida
        self.velocidade_total = 0.0      # soma das velocidades para calcular média
        self.pontos_acumulados = 0.0     # fitness acumulado em tempo real
        self.indice_carro = indice
        self.total_carros = total_carros
        
    def reset(self):
        """Recomeça a tentativa"""
        # Recalcula posição baseada na linha de largada
        x, y, angulo = calcular_posicao_inicial_carro(self.indice_carro, self.total_carros)
        self.x = x
        self.y = y
        self.velocidade = CONFIG['carros']['velocidade_inicial']
        self.angulo = angulo
        self.vivo = True
        self.distancia_percorrida = 0.0
        self.tempo_vivo = 0
        self.trajetoria_x = [self.x]
        self.trajetoria_y = [self.y]
        self.voltas_completas = 0
        self.distancia_desde_ultima_volta = 0.0
        self.frame_inicio_volta = 0
        self.melhor_tempo_volta = None
        self.velocidade_total = 0.0
        self.pontos_acumulados = 0.0
        self.max_afastamento_chegada = 0.0
        self.max_afastamento_chegada = 0.0  # maximo afastamento euclideano da linha de chegada desde o ultimo cruzamento
        
    def get_sensores(self, pista_config):
        """5 sensores: -90, -45, 0, +45, +90 graus em relacao ao angulo do carro"""
        sensores = [15.0] * 5
        angulos = [
            self.angulo - 90,
            self.angulo - 45,
            self.angulo,
            self.angulo + 45,
            self.angulo + 90,
        ]
        
        cx = pista_config['centro_x']
        cy = pista_config['centro_y']
        metade_reta = pista_config['metade_reta']
        raio_ext = pista_config['raio_externo']
        raio_int = pista_config['raio_interno']
        
        for i, ang in enumerate(angulos):
            rad = np.radians(ang)
            for dist in np.arange(0.2, 15, 0.2):
                sx = self.x + dist * np.cos(rad)
                sy = self.y + dist * np.sin(rad)
                # Fora da pista: fora da borda externa OU dentro da ilha interna
                if not ponto_dentro_estadio(sx, sy, cx, cy, metade_reta, raio_ext) \
                   or ponto_dentro_estadio(sx, sy, cx, cy, metade_reta, raio_int):
                    sensores[i] = dist
                    break
        
        return np.array(sensores) / 15.0
    
    def mover(self, acao):
        """Move o carrinho baseado na ação da rede neural"""
        if not self.vivo:
            return
        
        # Armazena se estava sobre a linha de CHEGADA antes de mover
        estava_sobre_linha = esta_sobre_linha_chegada(self.x, self.y)
            
        self.angulo += acao[0] * CONFIG['carros']['angulo_virada_max']
        self.velocidade = CONFIG['carros']['velocidade_min'] + acao[1] * (CONFIG['carros']['velocidade_max'] - CONFIG['carros']['velocidade_min'])
        
        rad = np.radians(self.angulo)
        self.x += self.velocidade * np.cos(rad)
        self.y += self.velocidade * np.sin(rad)
        
        # Detecta cruzamento da linha de CHEGADA
        esta_sobre_linha = esta_sobre_linha_chegada(self.x, self.y)
        
        # Conta volta apenas quando CRUZA a linha de chegada (entra nela vindo de fora)
        # e o carro esteve longe o suficiente da linha (nao estava girando no lugar)
        if not estava_sobre_linha and esta_sobre_linha:
            if self.max_afastamento_chegada >= CONFIG['deteccao_volta']['afastamento_minimo_da_chegada']:
                # Calcula tempo desta volta em frames
                tempo_desta_volta = self.tempo_vivo - self.frame_inicio_volta
                # Verifica recorde ANTES de atualizar melhor_tempo_volta
                eh_recorde = (self.melhor_tempo_volta is None or tempo_desta_volta < self.melhor_tempo_volta)
                melhoria_frames = (self.melhor_tempo_volta - tempo_desta_volta) if (eh_recorde and self.melhor_tempo_volta is not None) else 0
                if eh_recorde:
                    self.melhor_tempo_volta = tempo_desta_volta
                self.frame_inicio_volta = self.tempo_vivo
                self.voltas_completas += 1
                self.distancia_desde_ultima_volta = 0.0
                self.max_afastamento_chegada = 0.0
                # Recompensa por completar volta
                rec = CONFIG['recompensas']
                self.pontos_acumulados += rec['recompensa_volta']
                # Bonus por tempo absoluto desta volta
                self.pontos_acumulados += max(0.0, rec['bonus_volta_rapida'] - tempo_desta_volta)
                # Bonus extra por bater o recorde pessoal (proporcional a melhoria)
                if melhoria_frames > 0:
                    self.pontos_acumulados += melhoria_frames * rec['bonus_melhoria_por_frame']
        
        self.trajetoria_x.append(self.x)
        self.trajetoria_y.append(self.y)
        
        self.distancia_percorrida += self.velocidade
        self.distancia_desde_ultima_volta += self.velocidade
        self.velocidade_total += self.velocidade
        self.tempo_vivo += 1
        # Atualiza maximo afastamento da linha de chegada (para anti-cheat de girar no lugar)
        lc = CONFIG['linha_chegada']
        dist_chegada = np.sqrt((self.x - lc['x'])**2 + (self.y - lc['y'])**2)
        if dist_chegada > self.max_afastamento_chegada:
            self.max_afastamento_chegada = dist_chegada
        # Recompensa por velocidade (quadratica): incentiva fortemente aumentar velocidade
        rec = CONFIG['recompensas']
        vel_norm = self.velocidade / CONFIG['carros']['velocidade_max']
        self.pontos_acumulados += (vel_norm ** 2) * rec['peso_velocidade']
        if self.velocidade < rec['velocidade_lenta_limiar']:
            self.pontos_acumulados -= rec['penalidade_devagar']
        
    def checar_colisao(self, pista_config):
        """Verifica se saiu da pista"""
        cx = pista_config['centro_x']
        cy = pista_config['centro_y']
        metade_reta = pista_config['metade_reta']
        raio_ext = pista_config['raio_externo']
        raio_int = pista_config['raio_interno']
        
        # Válido: dentro do extádion externo E fora do estádion interno
        if not ponto_dentro_estadio(self.x, self.y, cx, cy, metade_reta, raio_ext) \
           or ponto_dentro_estadio(self.x, self.y, cx, cy, metade_reta, raio_int):
            self.vivo = False
            self.pontos_acumulados -= CONFIG['recompensas']['penalidade_colisao']
            return True
        return False


class RedeNeuralCarrinho:
    """Rede Neural que controla o carrinho — 5 entradas (sensores), 10 ocultos, 2 saidas"""
    
    def __init__(self):
        self.W1 = np.random.randn(5, 10) * 0.5
        self.b1 = np.zeros((1, 10))
        self.W2 = np.random.randn(10, 2) * 0.5
        self.b2 = np.zeros((1, 2))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def decidir_acao(self, sensores):
        """Recebe dados dos sensores e decide o que fazer"""
        sensores = sensores.reshape(1, -1)
        z1 = np.dot(sensores, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        virar = (a2[0, 0] - 0.5) * 2
        acelerar = a2[0, 1]
        
        return np.array([virar, acelerar])
    
    def mutar(self, taxa=0.1):
        """Faz pequenas mudanças aleatórias nos pesos"""
        self.W1 += np.random.randn(*self.W1.shape) * taxa
        self.W2 += np.random.randn(*self.W2.shape) * taxa
        self.b1 += np.random.randn(*self.b1.shape) * taxa
        self.b2 += np.random.randn(*self.b2.shape) * taxa
    
    def copiar_de(self, outra_rede):
        """Copia os pesos de outra rede"""
        self.W1 = outra_rede.W1.copy()
        self.b1 = outra_rede.b1.copy()
        self.W2 = outra_rede.W2.copy()
        self.b2 = outra_rede.b2.copy()


def criar_pista():
    """
    Define os limites da pista oval
    Retorna bordas interna e externa
    """
    return CONFIG['pista']


class SimuladorAprendizado:
    """Simula e visualiza o aprendizado em tempo real"""
    
    def __init__(self):
        self.pista_config = criar_pista()
        self.populacao_size = CONFIG['simulacao']['populacao']
        self.carrinhos = [CarrinhoIA(i, self.populacao_size) for i in range(self.populacao_size)]
        self.cerebros = [RedeNeuralCarrinho() for _ in range(self.populacao_size)]
        
        self.geracao = 0
        self.frame_atual = 0
        self.max_frames_por_tentativa = CONFIG['simulacao']['max_frames_tentativa']
        
        self.historico_melhor = []
        self.historico_media = []
        self.melhor_fitness_geral = 0
        self.vencedor = None
        self.parar_animacao = False
        self._bg = None  # canvas background cache for blit
        self._frame_count = 0
        
        # Configuração da visualização - pista em cima (largura toda), gráfico abaixo
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.suptitle('IA Aprendendo a Pilotar - Circuito Oval', 
                         fontsize=14, fontweight='bold')
        
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.35)
        self.ax_pista = self.fig.add_subplot(gs[0])
        self.ax_stats = self.fig.add_subplot(gs[1])

        # Pre-computa geometria da pista (feito uma única vez)
        self._precomputar_geometria_pista()

        # Conectar eventos para fechar
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)        
        self.iniciar_tentativa()
    
    def on_key_press(self, event):
        """Handler para teclas pressionadas"""
        if event.key == 'q':
            print("\nParando simulação...")
            self.parar_animacao = True
            plt.close(self.fig)
    
    def on_close(self, event):
        """Handler para fechamento da janela"""
        print("\nJanela fechada. Parando simulação...")
        self.parar_animacao = True
        
    def _precomputar_geometria_pista(self):
        """Pré-computa os contornos da pista e as zebras de largada/chegada (executa uma única vez)."""
        p = CONFIG['pista']
        cx, cy, ml = p['centro_x'], p['centro_y'], p['metade_reta']
        self._ext_x, self._ext_y = gerar_contorno_estadio(cx, cy, ml, p['raio_externo'])
        self._int_x, self._int_y = gerar_contorno_estadio(cx, cy, ml, p['raio_interno'])

        # Pré-calcula os patches das zebras (lista de (x, y, w, h, cor, transform_args))
        self._zebras_largada = self._calcular_zebras(
            CONFIG['linha_largada'], cores=('white', 'black'))
        self._zebras_chegada = self._calcular_zebras(
            CONFIG['linha_chegada'], cores=('white', '#e74c3c'))

    def _calcular_zebras(self, cfg, cores):
        """Retorna lista de dicts com parâmetros de cada retângulo zebra pré-calculados."""
        x_centro = cfg['x']
        y_centro = cfg['y']
        largura  = cfg['largura']
        altura   = cfg['altura']
        angulo   = cfg['angulo']
        n        = cfg['num_zebras']
        w_zebra  = largura / n
        x_start  = x_centro - largura / 2
        zebras = []
        for i in range(n):
            cor = cores[i % 2]
            zebras.append(dict(
                xy=(x_start + i * w_zebra, y_centro - altura / 2),
                width=w_zebra, height=altura,
                cor=cor,
                cx=x_centro, cy=y_centro, angulo=angulo
            ))
        return zebras

    def _preparar_fundo(self):
        """Desenha elementos estáticos (pista + lines + stats) e salva o background para blit."""
        # --- PISTA ---
        ax = self.ax_pista
        ax.clear()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_title(f'Circuito - Tentativa {self.geracao + 1} (Frame --)',
                     fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.set_facecolor('#1e8449')
        ax.set_aspect('auto')

        ax.fill(self._ext_x, self._ext_y, color='#7f8c8d', zorder=1)
        ax.fill(self._int_x, self._int_y, color='#1e8449', zorder=2)
        ax.plot(self._ext_x, self._ext_y, 'w-', linewidth=4, zorder=3)
        ax.plot(self._int_x, self._int_y, 'w-', linewidth=4, zorder=3)

        for z in self._zebras_largada:
            t = Affine2D().rotate_deg_around(z['cx'], z['cy'], z['angulo']) + ax.transData
            p = Rectangle(z['xy'], z['width'], z['height'],
                          facecolor=z['cor'], edgecolor='none', zorder=5)
            p.set_transform(t)
            ax.add_patch(p)

        for z in self._zebras_chegada:
            t = Affine2D().rotate_deg_around(z['cx'], z['cy'], z['angulo']) + ax.transData
            p = Rectangle(z['xy'], z['width'], z['height'],
                          facecolor=z['cor'], edgecolor='none', zorder=5)
            p.set_transform(t)
            ax.add_patch(p)

        # --- STATS ---
        self._desenhar_stats()

        plt.tight_layout()
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _desenhar_stats(self):
        """Desenha o gráfico de evolução e o painel de status (chamado apenas no fundo estático)."""
        ax = self.ax_stats
        ax.clear()
        if len(self.historico_melhor) > 0:
            ax.plot(self.historico_melhor, label='Melhor',
                    color=CONFIG['carros']['cor_melhor'], linewidth=3)
            ax.plot(self.historico_media, label='Media',
                    color=CONFIG['carros']['cor_normal'], linewidth=2)
            ax.set_xlabel('Tentativa (Geração)', fontsize=11)
            ax.set_ylabel('Fitness', fontsize=11)
            ax.set_title('Evolução do Aprendizado', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=self.melhor_fitness_geral, color='r',
                       linestyle='--', linewidth=1, alpha=0.5)
            ax.text(len(self.historico_melhor) - 1, self.melhor_fitness_geral,
                    f' Recorde: {self.melhor_fitness_geral:.1f}',
                    fontsize=9, color='red', va='bottom')

        cor_label = CONFIG['carros']['cor_normal']
        status  = f"TENTATIVA #{self.geracao + 1}\n\n"
        status += f"Objetivo: {CONFIG['simulacao']['voltas_objetivo']} VOLTAS\n"
        status += f"• Ficar DENTRO da pista\n"
        status += f"• Completar as voltas\n\n"
        status += f"Como aprende:\n"
        status += f"• 3 sensores detectam bordas\n"
        status += f"• IA decide direcao\n"
        status += f"• Melhores evoluem\n\n"
        status += f"Verde  = Melhor\n"
        status += f"Normal = Outros\n"
        status += f"X      = Saiu"
        ax.text(0.02, 0.02, status, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9),
                family='monospace')

    def iniciar_tentativa(self):
        """Inicia uma nova tentativa (geração)"""
        for c in self.carrinhos:
            c.reset()
        self.frame_atual = 0
        
    def simular_frame(self):
        """Simula um frame da simulação"""
        vivos = 0
        for i, (carrinho, cerebro) in enumerate(zip(self.carrinhos, self.cerebros)):
            if not carrinho.vivo:
                continue            
            # Verifica se completou o objetivo de voltas
            if carrinho.voltas_completas >= CONFIG['simulacao']['voltas_objetivo']:
                self.vencedor = i
                return True                
            vivos += 1
            sensores = carrinho.get_sensores(self.pista_config)
            # Kill por estagnacao: nao progrediu o suficiente nos primeiros N frames
            cfg_sim = CONFIG['simulacao']
            if (carrinho.voltas_completas == 0
                    and carrinho.tempo_vivo >= cfg_sim['frames_limite_progresso']
                    and carrinho.max_afastamento_chegada < cfg_sim['afastamento_minimo_progresso']):
                carrinho.vivo = False
                carrinho.pontos_acumulados -= CONFIG['recompensas']['penalidade_colisao']
                continue
            # Recompensa em tempo real por estar bem dentro da pista (sensor minimo = distancia normalizada da borda mais proxima)
            min_sensor = float(np.min(sensores))
            # Recompensa por centro ponderada pela velocidade: parado no centro nao vale nada
            vel_norm = carrinho.velocidade / CONFIG['carros']['velocidade_max']
            carrinho.pontos_acumulados += min_sensor * vel_norm * CONFIG['recompensas']['peso_centro']
            # Penalidade por contramao: compara direcao real com tangente esperada da pista
            rec = CONFIG['recompensas']
            rad = np.radians(carrinho.angulo)
            vel_dir = np.array([np.cos(rad), np.sin(rad)])
            tangente = calcular_tangente_pista(carrinho.x, carrinho.y, self.pista_config)
            dot = float(np.dot(vel_dir, tangente))
            if dot < rec['contramao_limiar']:
                carrinho.pontos_acumulados -= rec['penalidade_contramao'] * (-dot)
            acao = cerebro.decidir_acao(sensores)
            carrinho.mover(acao)
            carrinho.checar_colisao(self.pista_config)
        
        self.frame_atual += 1
        
        # Acabou a tentativa?
        if vivos == 0 or self.frame_atual >= self.max_frames_por_tentativa:
            return True  # Tentativa acabou
        return False  # Continua
    
    def evoluir_geracao(self):
        """Evolui para a proxima geracao usando fitness acumulado em tempo real."""
        fitness = [c.pontos_acumulados for c in self.carrinhos]
        
        melhor_fitness = max(fitness)
        media_fitness = np.mean(fitness)
        
        self.historico_melhor.append(melhor_fitness)
        self.historico_media.append(media_fitness)
        
        if melhor_fitness > self.melhor_fitness_geral:
            self.melhor_fitness_geral = melhor_fitness
        
        cfg = CONFIG['simulacao']
        top_n = cfg['top_sobreviventes']
        n_aleatorios = cfg['novos_aleatorios_por_geracao']

        # Selecao: top-N sobrevivem
        indices_ordenados = np.argsort(fitness)[::-1]
        melhores = indices_ordenados[:top_n]

        # Taxa de mutacao adaptativa: alta se nao melhorou nas ultimas 10 geracoes
        if len(self.historico_melhor) >= 10 and self.historico_melhor[-1] <= self.historico_melhor[-10] * 1.01:
            taxa_mutacao = 0.4  # platô detectado: explora mais
        elif self.geracao < 10:
            taxa_mutacao = 0.5  # inicio: explora bastante
        elif self.geracao < 30:
            taxa_mutacao = 0.3
        else:
            taxa_mutacao = 0.15

        novos_cerebros = []

        # Mantém os top-N exatos (sem mutar)
        for idx in melhores:
            novo = RedeNeuralCarrinho()
            novo.copiar_de(self.cerebros[idx])
            novos_cerebros.append(novo)

        # Injeta N completamente novos (diversidade)
        for _ in range(n_aleatorios):
            novos_cerebros.append(RedeNeuralCarrinho())

        # Resto: mutacoes dos melhores
        while len(novos_cerebros) < self.populacao_size:
            pai_idx = melhores[np.random.randint(0, min(3, len(melhores)))]
            filho = RedeNeuralCarrinho()
            filho.copiar_de(self.cerebros[pai_idx])
            filho.mutar(taxa_mutacao)
            novos_cerebros.append(filho)
        
        self.cerebros = novos_cerebros
        self.geracao += 1
        
    def atualizar_visualizacao(self):
        """Chamado pelo timer a cada frame — usa blit para renderizar só os carros."""
        if self.parar_animacao:
            return

        if self.vencedor is not None:
            print(f"\n" + "="*70)
            print(f"SUCESSO! Carro #{self.vencedor + 1} completou as voltas!")
            print(f"Geracao: {self.geracao}")
            print(f"="*70)
            plt.close()
            return

        # Simula N passos por frame (configurável)
        passos = CONFIG['simulacao'].get('passos_por_frame', 1)
        tentativa_acabou = False
        for _ in range(passos):
            tentativa_acabou = self.simular_frame()
            if tentativa_acabou:
                break

        if tentativa_acabou and self.vencedor is None:
            self.evoluir_geracao()
            self.iniciar_tentativa()
            # Nova geração: redesenha fundo estático com stats atualizados
            self._preparar_fundo()
            return

        # --- BLIT: restaura fundo estático ---
        self.fig.canvas.restore_region(self._bg)

        # --- Atualiza título com frame atual ---
        self.ax_pista.set_title(
            f'Circuito - Tentativa {self.geracao + 1} (Frame {self.frame_atual})',
            fontsize=12, fontweight='bold')
        self.ax_pista.draw_artist(self.ax_pista.title)

        # --- Desenha carros dinamicamente ---
        fitness_atual = []
        vivos = 0
        melhor_idx = 0

        for i, carrinho in enumerate(self.carrinhos):
            if carrinho.vivo:
                vivos += 1
                cor = CONFIG['carros']['cor_normal']
                alpha = 0.6
            else:
                cor = CONFIG['carros']['cor_morto']
                alpha = 0.2
            self._desenhar_carrinho_blit(carrinho, cor, alpha)
            fitness_atual.append(carrinho.pontos_acumulados)

        if fitness_atual:
            melhor_idx = int(np.argmax(fitness_atual))
            self._desenhar_carrinho_blit(
                self.carrinhos[melhor_idx], CONFIG['carros']['cor_melhor'], 1.0)

        # --- Info text ---
        tempos_volta_geral = [c.melhor_tempo_volta for c in self.carrinhos
                              if c.melhor_tempo_volta is not None]
        melhor_tempo_geral = min(tempos_volta_geral) if tempos_volta_geral else None

        # Painel geral (canto superior esquerdo)
        info_geral = (f'Geracao: {self.geracao + 1}\n'
                      f'Vivos: {vivos}/{self.populacao_size}\n'
                      f'Melhor volta: {melhor_tempo_geral if melhor_tempo_geral else "-"} fr')
        txt_geral = self.ax_pista.text(
            0.02, 0.98, info_geral, transform=self.ax_pista.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=20)
        self.ax_pista.draw_artist(txt_geral)
        txt_geral.remove()

        # Painel do melhor carro atual (canto superior direito)
        melhor = self.carrinhos[melhor_idx]
        vel_melhor = melhor.velocidade_total / melhor.tempo_vivo if melhor.tempo_vivo > 0 else 0
        status_melhor = 'VIVO' if melhor.vivo else 'morto'
        info_melhor = (f'[ Melhor atual | {status_melhor} ]\n'
                       f'Pontos: {melhor.pontos_acumulados:.0f}\n'
                       f'Voltas: {melhor.voltas_completas}/{CONFIG["simulacao"]["voltas_objetivo"]}\n'
                       f'Vel media: {vel_melhor:.2f}\n'
                       f'Melhor volta: {melhor.melhor_tempo_volta if melhor.melhor_tempo_volta else "-"} fr')
        txt_melhor = self.ax_pista.text(
            0.98, 0.98, info_melhor, transform=self.ax_pista.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.85),
            zorder=20)
        self.ax_pista.draw_artist(txt_melhor)
        txt_melhor.remove()

        # --- Blit final ---
        self.fig.canvas.blit(self.fig.bbox)

    def _desenhar_carrinho_blit(self, carrinho, cor, alpha):
        """Desenha um carrinho direto no renderer (sem ax.clear) via draw_artist."""
        ax = self.ax_pista

        if carrinho.vivo:
            tamanho = CONFIG['carros']['tamanho']
            rad = np.radians(carrinho.angulo)
            dx  = tamanho * np.cos(rad)
            dy  = tamanho * np.sin(rad)
            dpx = tamanho * 0.5 * np.cos(rad + np.pi / 2)
            dpy = tamanho * 0.5 * np.sin(rad + np.pi / 2)
            vertices = [
                [carrinho.x + dx,          carrinho.y + dy],
                [carrinho.x - dx + dpx,    carrinho.y - dy + dpy],
                [carrinho.x - dx - dpx,    carrinho.y - dy - dpy],
            ]
            tri = Polygon(vertices, closed=True, color=cor,
                          edgecolor='black', linewidth=2, alpha=alpha, zorder=10)
            ax.add_patch(tri)
            ax.draw_artist(tri)
            tri.remove()

            # Sensores (5: lateral esq, diagonal esq, frente, diagonal dir, lateral dir)
            sensores = carrinho.get_sensores(self.pista_config)
            if CONFIG['visualizacao']['mostrar_sensores']:
                angulos_sensor = [carrinho.angulo - 90, carrinho.angulo - 45, carrinho.angulo,
                                  carrinho.angulo + 45, carrinho.angulo + 90]
                cores_sensor = CONFIG['visualizacao']['cores_sensores']
                for ang, dist, cs in zip(angulos_sensor, sensores * 15, cores_sensor):
                    r = np.radians(ang)
                    ex = carrinho.x + dist * np.cos(r)
                    ey = carrinho.y + dist * np.sin(r)
                    sl = Line2D([carrinho.x, ex], [carrinho.y, ey], linestyle='--',
                                color=cs, alpha=alpha * 0.5, linewidth=1.5)
                    ax.add_line(sl)
                    ax.draw_artist(sl)
                    sl.remove()
        else:
            mk = Line2D([carrinho.x], [carrinho.y], marker='X', color='red',
                        markersize=12, markeredgewidth=2, alpha=alpha, linestyle='none')
            ax.add_line(mk)
            ax.draw_artist(mk)
            mk.remove()

    def iniciar_animacao(self):
        """Inicia a animação usando timer para controle total do blit."""
        print("=" * 60)
        print("IA APRENDENDO A PILOTAR - CIRCUITO OVAL")
        print("=" * 60)
        print(f"\nOBJETIVO: {CONFIG['simulacao']['voltas_objetivo']} voltas sem sair da pista")
        print("• CINZA = Pista (area valida)")
        print("• VERDE = Fora da pista")
        print("• Pressione 'Q' ou feche a janela para parar.")
        print("=" * 60)

        # Prepara fundo estático uma vez (desenha pista + stats + salva background)
        self._preparar_fundo()

        interval = CONFIG['simulacao']['fps_interval_ms']
        timer = self.fig.canvas.new_timer(interval=interval)
        timer.add_callback(self.atualizar_visualizacao)
        timer.start()
        plt.show()


if __name__ == "__main__":
    simulador = SimuladorAprendizado()
    simulador.iniciar_animacao()
