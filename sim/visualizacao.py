# Camada de visualização — toda a renderização via matplotlib.
# SimuladorAprendizado estende SimuladorBase com a janela e o loop em tempo real.

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mc
from matplotlib.patches import Rectangle, Polygon, PathPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.collections import PolyCollection, LineCollection

from .track import (
    CONFIG, _COR, _CENTERLINE, _CL_X, _CL_Y,
    _TRACK_XLIM, _TRACK_YLIM, _NOME_PISTA,
    gerar_contornos_pista,
)
from .simulacao import SimuladorBase


class SimuladorAprendizado(SimuladorBase):
    """SimuladorBase + visualização matplotlib em tempo real (blit)."""

    def __init__(self):
        super().__init__()

        # UI state
        self._bg = None
        self._bg_refresh_pending = False
        self._frame_count = 0
        self._notificacoes = []
        self._melhor_idx_ant = None
        self._help_visible = False
        self._log_decisoes = []
        self._MAX_LOG = 13
        self._acao_atual = np.array([0.0, 0.5])

        # Figure layout
        self.fig = plt.figure(figsize=(16, 11), facecolor=_COR['cor_card'])
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
        self.ax_pista = self.fig.add_subplot(gs[0])

        c = CONFIG.get('cards', {})
        wr = [
            c['size_grafico'],
            c['size_info'],
            c['size_comandos'],
            c['size_eventos'],
            c['size_rede_neural'],
        ]
        gs_cards = self.fig.add_gridspec(
            1, 5, width_ratios=wr, wspace=c['wspace'],
            left=0.03, right=0.99, top=0.305, bottom=0.08,
        )
        self.ax_stats = self.fig.add_subplot(gs_cards[0])
        self.ax_info = self.fig.add_subplot(gs_cards[1])
        self.ax_log = self.fig.add_subplot(gs_cards[2])
        self.ax_eventos = self.fig.add_subplot(gs_cards[3])
        self.ax_rede = self.fig.add_subplot(gs_cards[4])

        self._precomputar_geometria_pista()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.iniciar_tentativa()

    # Eventos de janela

    def on_key_press(self, event):
        if event.key == 'q':
            print("\nStopping simulation...")
            self.parar_animacao = True
            plt.close(self.fig)

    def on_close(self, event):
        print("\nWindow closed. Stopping simulation...")
        self.parar_animacao = True

    def on_mouse_click(self, event):
        """Detecta clique no botão ? e abre/fecha o overlay de ajuda."""
        if event.inaxes != self.ax_pista:
            return
        ax = self.ax_pista
        disp = ax.transAxes.transform([[0.008, 0.022], [0.030, 0.054]])
        x0, y0 = disp[0]
        x1, y1 = disp[1]
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            self._help_visible = not self._help_visible

    # Geometria da pista pré-computada (executado uma única vez)

    def _precomputar_geometria_pista(self):
        """Calcula contornos, compound path e zebras (executa uma única vez)."""
        ext_x, ext_y, int_x, int_y = gerar_contornos_pista()
        N, M = len(ext_x), len(int_x)
        outer = np.column_stack([ext_x, ext_y])
        inner = np.column_stack([int_x[::-1], int_y[::-1]])
        verts = np.concatenate([outer, outer[0:1], inner, inner[0:1]])
        codes = np.array(
            [MplPath.MOVETO] + [MplPath.LINETO] * (N - 1) + [MplPath.CLOSEPOLY] +
            [MplPath.MOVETO] + [MplPath.LINETO] * (M - 1) + [MplPath.CLOSEPOLY],
            dtype=np.uint8,
        )
        self._track_path = MplPath(verts, codes)
        self._ext_x = np.append(ext_x, ext_x[0])
        self._ext_y = np.append(ext_y, ext_y[0])
        self._int_x = np.append(int_x, int_x[0])
        self._int_y = np.append(int_y, int_y[0])
        self._cl_cx = np.append(_CENTERLINE[:, 0], _CENTERLINE[0, 0])
        self._cl_cy = np.append(_CENTERLINE[:, 1], _CENTERLINE[0, 1])

        self._zebras_linha = self._calcular_zebras(
            CONFIG['linha_largada'], cores=(_COR['branco'], _COR['preto']))

    def _calcular_zebras(self, cfg, cores):
        x_centro = cfg['x'];   y_centro = cfg['y']
        largura  = cfg['largura'];  altura  = cfg['altura']
        angulo = cfg['angulo'];  n = cfg['num_zebras']
        w_zebra = largura / n;  x_start = x_centro - largura / 2
        zebras = []
        for i in range(n):
            zebras.append(dict(
                xy=(x_start + i * w_zebra, y_centro - altura / 2),
                width=w_zebra, height=altura,
                cor=cores[i % 2],
                cx=x_centro, cy=y_centro, angulo=angulo,
            ))
        return zebras

    # Fundo estático (pista + painéis). Salvo em self._bg para blit.

    def _preparar_fundo(self):
        """Desenha elementos estáticos e salva o background para blit."""
        ax = self.ax_pista
        ax.clear()
        ax.set_xlim(*_TRACK_XLIM)
        ax.set_ylim(*_TRACK_YLIM)
        ax.grid(False)
        ax.set_facecolor(_COR['cor_grama'])
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.add_patch(PathPatch(self._track_path,
                               facecolor=_COR['cor_asfalto'], edgecolor='none', zorder=1))
        ax.plot(self._ext_x, self._ext_y, color=_COR['branco'], linewidth=1.5, zorder=3)
        ax.plot(self._int_x, self._int_y, color=_COR['branco'], linewidth=1.5, zorder=3)

        ax.text(0.5, 0.978, _NOME_PISTA.upper(), transform=ax.transAxes,
                fontsize=9, fontweight='bold', color=_COR['texto_principal'],
                va='top', ha='center', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=_COR['cor_card'],
                          edgecolor=_COR['cinza_medio'], alpha=0.75), zorder=10)

        ax.plot(self._cl_cx, self._cl_cy,
                linestyle=(0, (6, 8)), color=_COR['branco'],
                linewidth=0.8, alpha=0.35, zorder=2)

        for z in self._zebras_linha:
            t = Affine2D().rotate_deg_around(z['cx'], z['cy'], z['angulo']) + ax.transData
            p = Rectangle(z['xy'], z['width'], z['height'],
                          facecolor=z['cor'], edgecolor='none', zorder=5)
            p.set_transform(t)
            ax.add_patch(p)

        self._desenhar_stats()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05, hspace=0.0)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    # Painéis estáticos (gráfico de evolução + cards)

    def _desenhar_stats(self):
        """Gráfico de evolução + cards de informação (chamado no fundo estático)."""

        # --- Card de arquitetura / runtime ---
        ai = self.ax_info
        ai.clear()
        ai.set_facecolor(_COR['cor_card'])
        ai.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ai.spines.values():
            spine.set_visible(False)
        ai.set_title('Simulation', fontsize=9, fontweight='bold',
                     color=_COR['texto_secundario'], pad=4)

        x_title = 0.08
        pop = CONFIG['simulacao']['populacao']
        voltas = CONFIG['simulacao']['voltas_objetivo']
        stag = self.gens_sem_melhora
        mut_pct = f'{self.taxa_mutacao_atual*100:.0f}%  [{self.modo_mutacao}]'
        avg_laps = f'{self.avg_laps_gen:.2f}'
        stag_cor = (_COR['verde']   if stag < 5  else
                    _COR['amarelo'] if stag < 15 else _COR['vermelho'])
        mut_cor  = (_COR['vermelho'] if self.modo_mutacao in ('CRITICAL', 'plateau') else
                    _COR['amarelo']  if self.modo_mutacao == 'shake'                 else
                    _COR['texto_secundario'])

        sections = [
            (0.95, 'ARCHITECTURE', None),
            (0.87, 'Inputs',       '7 sensors + speed'),
            (0.80, 'Layers',       '8 → 14 → 2'),
            (0.73, 'Activation',   'sigmoid'),
            (None, None,           None),
            (0.62, 'TRAINING',     None),
            (0.54, 'Population',   f'{pop} agents'),
            (0.47, 'Objective',    f'{voltas} laps'),
            (None, None,           None),
            (0.36, 'RUNTIME',      None),
            (0.28, 'Stagnation',   (f'{stag} gens',  stag_cor)),
            (0.21, 'Avg laps',     (avg_laps,         _COR['texto_secundario'])),
            (0.14, 'Mutation',     (mut_pct,          mut_cor)),
        ]

        for y, label, value in sections:
            if label is None:
                continue
            if value is None:
                if y < 0.93:
                    ai.plot([0.05, 0.95], [y + 0.04, y + 0.04],
                            color=_COR['cinza_inativo'], linewidth=0.6,
                            transform=ai.transAxes, clip_on=True)
                ai.text(x_title, y, label, transform=ai.transAxes,
                        fontsize=7.5, verticalalignment='top', horizontalalignment='left',
                        family='monospace', color=_COR['cinza_medio'], fontweight='bold')
            elif isinstance(value, tuple):
                val_str, val_cor = value
                ai.text(x_title, y, label, transform=ai.transAxes,
                        fontsize=7.5, verticalalignment='top', horizontalalignment='left',
                        family='monospace', color=_COR['texto_terciario'])
                ai.text(0.98, y, val_str, transform=ai.transAxes,
                        fontsize=7.5, verticalalignment='top', horizontalalignment='right',
                        family='monospace', color=val_cor, fontweight='bold')
            else:
                ai.text(x_title, y, label, transform=ai.transAxes,
                        fontsize=7.5, verticalalignment='top', horizontalalignment='left',
                        family='monospace', color=_COR['texto_terciario'])
                ai.text(0.98, y, value, transform=ai.transAxes,
                        fontsize=7.5, verticalalignment='top', horizontalalignment='right',
                        family='monospace', color=_COR['texto_secundario'])

        # --- Gráfico de evolução ---
        _JANELA = 200
        ax = self.ax_stats
        ax.clear()
        ax.set_facecolor(_COR['cor_card'])
        for spine in ax.spines.values():
            spine.set_edgecolor(_COR['cinza_medio'])
        ax.tick_params(colors=_COR['texto_terciario'], labelsize=7.5)
        if self.historico_melhor:
            total = len(self.historico_melhor)
            inicio = max(0, total - _JANELA)
            gens = list(range(inicio, total))
            ax.plot(gens, self.historico_melhor[inicio:], label='Best',
                    color=_COR['cor_melhor'], linewidth=3)
            ax.plot(gens, self.historico_media[inicio:],  label='Average',
                    color=_COR['cor_normal'],  linewidth=2)
            ax.set_xlim(inicio, inicio + _JANELA - 1)
            ax.set_xlabel('Generation', fontsize=8, color=_COR['texto_secundario'])
            ax.set_title('Learning Progress', fontsize=9, fontweight='bold',
                         color=_COR['texto_secundario'], pad=4)
            leg = ax.legend(loc='upper left', fontsize=8)
            for txt in leg.get_texts():
                txt.set_color(_COR['texto_secundario'])
            leg.get_frame().set_facecolor(_COR['cor_card'])
            leg.get_frame().set_edgecolor(_COR['cinza_medio'])
            ax.grid(True, alpha=0.15, color=_COR['cinza_medio'])

        # Setup ax_log
        al = self.ax_log
        al.clear()
        al.set_facecolor(_COR['cor_card'])
        al.set_title('AI Commands', fontsize=9, fontweight='bold',
                     color=_COR['texto_secundario'], pad=4)
        al.set_xlim(0, 10);  al.set_ylim(0, 10)
        al.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in al.spines.values():
            spine.set_visible(False)

        # Setup ax_eventos
        ae = self.ax_eventos
        ae.clear()
        ae.set_facecolor(_COR['cor_card'])
        ae.set_title('Events', fontsize=9, fontweight='bold',
                     color=_COR['texto_secundario'], pad=4)
        ae.set_xlim(0, 10);  ae.set_ylim(0, 10)
        ae.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ae.spines.values():
            spine.set_visible(False)

        # Setup ax_rede
        ar = self.ax_rede
        ar.clear()
        ar.set_facecolor(_COR['cor_card'])
        ar.set_xlim(0, 1);  ar.set_ylim(0, 1)
        ar.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ar.spines.values():
            spine.set_visible(False)
        ar.set_title('Neural Network', fontsize=9, fontweight='bold',
                     color=_COR['texto_secundario'], pad=4)

    # Visualização da rede neural (blit)

    def _desenhar_rede_neural(self, melhor_idx):
        an = self.ax_rede
        if melhor_idx is None:
            return
        carr = self.carrinhos[melhor_idx]
        cerebro = self.cerebros[melhor_idx]

        s  = np.clip(getattr(carr, '_cached_sensores', np.zeros(8)), 0.0, 1.0)
        a1 = 1.0 / (1.0 + np.exp(-np.clip(s @ cerebro.W1 + cerebro.b1, -500, 500)))
        a2 = 1.0 / (1.0 + np.exp(-np.clip(a1 @ cerebro.W2 + cerebro.b2, -500, 500)))

        x_in, x_hid, x_out = 0.20, 0.52, 0.74
        in_ys  = np.linspace(0.87, 0.08, 8)
        hid_ys = np.linspace(0.87, 0.05, 14)
        out_ys = np.array([0.65, 0.30])

        for x, lbl in [(x_in, 'IN'), (x_hid, 'HIDDEN'), (x_out, 'OUT')]:
            t = an.text(x, 0.99, lbl, transform=an.transAxes, fontsize=5.5,
                        color=_COR['cinza_medio'], va='top', ha='center',
                        fontweight='bold', zorder=6)
            an.draw_artist(t); t.remove()

        c_pos = _mc.to_rgba(_COR['verde'])
        c_neg = _mc.to_rgba(_COR['vermelho'])

        # W1 connections
        w1_max = max(np.abs(cerebro.W1).max(), 1e-6)
        segs, cols, lws = [], [], []
        for i in range(8):
            for j in range(14):
                w  = cerebro.W1[i, j]
                aw = abs(w) / w1_max
                if aw < 0.15:
                    continue
                segs.append([(x_in, in_ys[i]), (x_hid, hid_ys[j])])
                base = c_pos if w > 0 else c_neg
                cols.append((base[0], base[1], base[2], aw * 0.45))
                lws.append(0.3 + aw * 0.6)
        if segs:
            lc = LineCollection(segs, colors=cols, linewidths=lws,
                                transform=an.transAxes, zorder=2)
            an.add_collection(lc); an.draw_artist(lc); lc.remove()

        # W2 connections
        w2_max = max(np.abs(cerebro.W2).max(), 1e-6)
        segs, cols, lws = [], [], []
        for j in range(14):
            for k in range(2):
                w  = cerebro.W2[j, k]
                aw = abs(w) / w2_max
                if aw < 0.10:
                    continue
                segs.append([(x_hid, hid_ys[j]), (x_out, out_ys[k])])
                base = c_pos if w > 0 else c_neg
                cols.append((base[0], base[1], base[2], aw * 0.60))
                lws.append(0.4 + aw * 0.9)
        if segs:
            lc = LineCollection(segs, colors=cols, linewidths=lws,
                                transform=an.transAxes, zorder=2)
            an.add_collection(lc); an.draw_artist(lc); lc.remove()

        # Input nodes
        c_in = _mc.to_rgba(_COR['cor_normal'])
        in_rgba = [(c_in[0], c_in[1], c_in[2], 0.20 + float(v)*0.80) for v in s]
        sc = an.scatter([x_in]*8, in_ys, s=28, c=in_rgba, edgecolors='none',
                        transform=an.transAxes, zorder=5, clip_on=False)
        an.draw_artist(sc); sc.remove()

        # Hidden nodes
        c_on  = _mc.to_rgba(_COR['cor_melhor'])
        c_off = _mc.to_rgba(_COR['cinza_inativo'])
        hid_rgba = []
        for v in a1:
            v = float(v)
            hid_rgba.append((c_on[0]*v + c_off[0]*(1-v),
                             c_on[1]*v + c_off[1]*(1-v),
                             c_on[2]*v + c_off[2]*(1-v),
                             0.25 + v*0.75))
        sc = an.scatter([x_hid]*14, hid_ys, s=18, c=hid_rgba, edgecolors='none',
                        transform=an.transAxes, zorder=5, clip_on=False)
        an.draw_artist(sc); sc.remove()

        # Output nodes
        steer = float((a2[0] - 0.5) * 2.0)
        gas_v = float(a2[1])
        esq_int = max(0.0, -steer)
        dir_int = max(0.0,  steer)
        fre_int = max(0.0, (0.5 - gas_v) * 2.0)
        gas_int = max(0.0, (gas_v - 0.5) * 2.0)
        c_ama = _mc.to_rgba(_COR['amarelo'])
        c_ver = _mc.to_rgba(_COR['verde'])
        c_vrm = _mc.to_rgba(_COR['vermelho'])
        sc_cols = [
            (c_ama[0], c_ama[1], c_ama[2], 0.25 + max(esq_int, dir_int)*0.75),
            (c_ver[0]*gas_int + c_vrm[0]*fre_int,
             c_ver[1]*gas_int + c_vrm[1]*fre_int,
             c_ver[2]*gas_int + c_vrm[2]*fre_int,
             0.25 + max(gas_int, fre_int)*0.75),
        ]
        sc = an.scatter([x_out]*2, out_ys, s=55, c=sc_cols, edgecolors='none',
                        transform=an.transAxes, zorder=5, clip_on=False)
        an.draw_artist(sc); sc.remove()

        # Input labels
        for y, lbl in zip(in_ys, ['-90', '-45', '-22', ' 0 ', '+22', '+45', '+90', 'vel']):
            t = an.text(0.01, y, lbl, transform=an.transAxes, fontsize=5.5,
                        color=_COR['texto_terciario'], va='center', ha='left',
                        family='monospace', zorder=6)
            an.draw_artist(t); t.remove()

        # Output labels
        for y_no, lbl, rgba, intensity, dy in [
            (out_ys[0], 'LFT', c_ama, esq_int, 0.10),
            (out_ys[0], 'RGT', c_ama, dir_int, 0.02),
            (out_ys[1], 'GAS', c_ver, gas_int, 0.10),
            (out_ys[1], 'BRK', c_vrm, fre_int, 0.02),
        ]:
            alpha = 0.25 + intensity * 0.75
            fw = 'bold' if intensity > 0.3 else 'normal'
            t = an.text(x_out+0.13, y_no+dy, lbl, transform=an.transAxes,
                        fontsize=5.5, fontweight=fw,
                        color=(rgba[0], rgba[1], rgba[2], alpha),
                        va='center', ha='left', zorder=6)
            an.draw_artist(t); t.remove()

        for y_no, val, rgba in [
            (out_ys[0], steer, c_ama),
            (out_ys[1], gas_v, c_ver if gas_v >= 0.5 else c_vrm),
        ]:
            tv = an.text(x_out+0.13, y_no-0.06, f'{val:+.2f}', transform=an.transAxes,
                         fontsize=6.5, fontweight='bold',
                         color=(rgba[0], rgba[1], rgba[2], 0.9),
                         va='center', ha='left', zorder=6)
            an.draw_artist(tv); tv.remove()

    # Carrinho individual (blit)

    def _desenhar_carrinho_blit(self, carrinho, cor, alpha):
        ax = self.ax_pista
        if carrinho.vivo:
            tamanho = CONFIG['carros']['tamanho']
            rad = np.radians(carrinho.angulo)
            dx  = tamanho * np.cos(rad)
            dy  = tamanho * np.sin(rad)
            dpx = tamanho * 0.5 * np.cos(rad + np.pi / 2)
            dpy = tamanho * 0.5 * np.sin(rad + np.pi / 2)
            vertices = [
                [carrinho.x + dx,       carrinho.y + dy],
                [carrinho.x - dx + dpx, carrinho.y - dy + dpy],
                [carrinho.x - dx - dpx, carrinho.y - dy - dpy],
            ]
            tri = Polygon(vertices, closed=True, color=cor,
                          edgecolor='none', linewidth=0, alpha=alpha, zorder=10)
            ax.add_patch(tri); ax.draw_artist(tri); tri.remove()

            if CONFIG['visualizacao']['mostrar_sensores']:
                sensores = getattr(carrinho, '_cached_sensores', None) or carrinho.get_sensores()
                angs = [carrinho.angulo + a for a in
                            [-90, -45, -22.5, 0, 22.5, 45, 90]]
                for ang, dist, cs in zip(angs, sensores[:7] * 15, _COR['cores_sensores']):
                    r  = np.radians(ang)
                    sl = Line2D([carrinho.x, carrinho.x + dist * np.cos(r)],
                                [carrinho.y, carrinho.y + dist * np.sin(r)],
                                linestyle='--', color=cs, alpha=alpha * 0.5, linewidth=1.5)
                    ax.add_line(sl); ax.draw_artist(sl); sl.remove()
        else:
            mk = Line2D([carrinho.x], [carrinho.y], marker='x',
                        color=_COR['cor_morto_scatter'],
                        markersize=5, markeredgewidth=1, alpha=0.5, linestyle='none')
            ax.add_line(mk); ax.draw_artist(mk); mk.remove()

    # Frame de atualização (chamado pelo timer)

    def atualizar_visualizacao(self):
        if self.parar_animacao:
            return
        if self._bg_refresh_pending:
            self._bg_refresh_pending = False
            self._preparar_fundo()
            return
        if self.vencedor is not None:
            self._finalizar_treinamento()
            return

        passos = CONFIG['simulacao']['passos_por_frame']
        tentativa_acabou = False
        for _ in range(passos):
            tentativa_acabou = self.simular_frame()
            if tentativa_acabou:
                break

        if tentativa_acabou and self.vencedor is None:
            self.evoluir_geracao()
            self.iniciar_tentativa()
            self._bg_refresh_pending = True
            return

        self._frame_count += 1
        self.fig.canvas.restore_region(self._bg)

        fitness_atual = [c.pontos_acumulados for c in self.carrinhos]
        vivos_mask = [c.vivo for c in self.carrinhos]
        vivos = sum(vivos_mask)
        indices_vivos = [i for i, v in enumerate(vivos_mask) if v]
        melhor_idx = (max(indices_vivos, key=lambda i: fitness_atual[i])
                      if indices_vivos else None)

        # Notificação por morte do melhor carro anterior
        if (self._melhor_idx_ant is not None
                and not self.carrinhos[self._melhor_idx_ant].vivo):
            c = self.carrinhos[self._melhor_idx_ant]
            self._notificacoes.append((time.time(), c.motivo_morte or 'Eliminated', c.pontos_acumulados))
            self._notificacoes = self._notificacoes[-1:]
        self._melhor_idx_ant = melhor_idx

        self._renderizar_carros(melhor_idx, vivos_mask)
        self._renderizar_overlays(melhor_idx, vivos)
        self._renderizar_painel_comandos(melhor_idx)
        self._renderizar_painel_eventos(melhor_idx)
        self._desenhar_rede_neural(melhor_idx)
        self.fig.canvas.blit(self.fig.bbox)

    def _desenhar_card_box(self, ax, lines, x, y, align='left',
                           pad_x=0.010, pad_y=0.010, line_gap=0.032, box_w=0.13):
        """Desenha um card com fundo arredondado e linhas de texto (blit-safe)."""
        box_h = pad_y * 2 + len(lines) * line_gap
        bx = x if align == 'left' else x - box_w
        by = y - box_h
        bg = FancyBboxPatch((bx, by), box_w, box_h,
                               boxstyle='round,pad=0.005',
                               facecolor=_COR['cor_card'], edgecolor=_COR['cinza_medio'],
                               linewidth=0.8, alpha=0.88,
                               transform=ax.transAxes, zorder=19, clip_on=False)
        ax.add_patch(bg); ax.draw_artist(bg); bg.remove()
        cur_y = y - pad_y
        for row in lines:
            if len(row) == 4:
                txt, fs, fw, color = row
                t = ax.text(bx + pad_x, cur_y, txt, transform=ax.transAxes,
                            fontsize=fs, verticalalignment='top', horizontalalignment='left',
                            fontweight=fw, color=color, zorder=21, family='monospace')
                ax.draw_artist(t); t.remove()
            else:
                label, valor, fs, fw, cor_label, cor_valor = row
                tl = ax.text(bx + pad_x, cur_y, label, transform=ax.transAxes,
                             fontsize=fs, verticalalignment='top', horizontalalignment='left',
                             fontweight=fw, color=cor_label, zorder=21, family='monospace')
                ax.draw_artist(tl); tl.remove()
                tr = ax.text(bx + box_w - pad_x, cur_y, valor, transform=ax.transAxes,
                             fontsize=fs, verticalalignment='top', horizontalalignment='right',
                             fontweight='normal', color=cor_valor, zorder=21, family='monospace')
                ax.draw_artist(tr); tr.remove()
            cur_y -= line_gap

    def _renderizar_carros(self, melhor_idx, vivos_mask):
        """Desenha carros mortos (×), carros vivos normais e o melhor carro."""
        ax = self.ax_pista
        sz = CONFIG['carros']['tamanho'] * 0.8

        dead_xs = [c.x for c, v in zip(self.carrinhos, vivos_mask) if not v]
        dead_ys = [c.y for c, v in zip(self.carrinhos, vivos_mask) if not v]
        if dead_xs:
            scat = ax.scatter(dead_xs, dead_ys, marker='x', c=_COR['cor_morto_scatter'],
                              s=CONFIG['carros']['morto_tamanho'],
                              linewidths=CONFIG['carros']['morto_espessura'],
                              alpha=0.3, zorder=9)
            ax.draw_artist(scat); scat.remove()

        live_verts, live_colors = [], []
        for i, carrinho in enumerate(self.carrinhos):
            if not carrinho.vivo or i == melhor_idx:
                continue
            rad = np.radians(carrinho.angulo)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            dx = sz * cos_a;  dy = sz * sin_a
            px = -sz * 0.5 * sin_a;  py = sz * 0.5 * cos_a
            live_verts.append([
                [carrinho.x + dx,       carrinho.y + dy],
                [carrinho.x - dx + px,  carrinho.y - dy + py],
                [carrinho.x - dx - px,  carrinho.y - dy - py],
            ])
            live_colors.append(_COR['cor_normal'])
        if live_verts:
            col = PolyCollection(live_verts, facecolors=live_colors,
                                 edgecolors='none', linewidths=0, alpha=1.0, zorder=10)
            ax.add_collection(col); ax.draw_artist(col); col.remove()

        if melhor_idx is not None:
            self._desenhar_carrinho_blit(self.carrinhos[melhor_idx], _COR['cor_melhor'], 1.0)

    def _renderizar_overlays(self, melhor_idx, vivos):
        """Desenha os cards de geração/melhor carro, botão de ajuda e notificações."""
        ax = self.ax_pista

        record_str = (f'{self.record_global_volta} fr / Gen. {self.record_global_volta_geracao}'
                      if self.record_global_volta else '---')
        self._desenhar_card_box(ax, [
            (f'GEN. {self.geracao + 1}',               11, 'bold',   _COR['texto_principal']),
            (f'Alive  {vivos}/{self.populacao_size}',    9, 'normal', _COR['texto_secundario']),
            (f'Record: {record_str}',                   9, 'normal', _COR['verde']),
        ], x=0.008, y=0.988, align='left', box_w=0.13)

        if melhor_idx is not None:
            melhor = self.carrinhos[melhor_idx]
            volta_str = f'{melhor.melhor_tempo_volta} fr' if melhor.melhor_tempo_volta else '---'
            fr_atual  = melhor.tempo_vivo - melhor.frame_inicio_volta
            if melhor.melhor_tempo_volta and fr_atual < melhor.melhor_tempo_volta:
                cor_fr = _COR['verde']
            elif melhor.voltas_completas > 0:
                cor_fr = _COR['amarelo']
            else:
                cor_fr = _COR['texto_secundario']
            self._desenhar_card_box(ax, [
                ('BEST ALIVE',                                                                    9, 'bold',   _COR['texto_principal']),
                ('Score',   f'{melhor.pontos_acumulados:.0f} pts',                               9, 'bold',   _COR['texto_secundario'], _COR['texto_principal']),
                ('Laps',    f'{melhor.voltas_completas}/{CONFIG["simulacao"]["voltas_objetivo"]}', 9, 'normal', _COR['texto_secundario'], _COR['texto_secundario']),
                ('Lap fr',  f'{fr_atual} fr',                                                    9, 'normal', _COR['texto_terciario'],   cor_fr),
                ('Best fr', volta_str,                                                            9, 'normal', _COR['texto_terciario'],   _COR['verde']),
            ], x=0.857, y=0.988, align='left', line_gap=0.028, box_w=0.135)

        # Botão ?
        hbx, hby, hbw, hbh = 0.008, 0.022, 0.022, 0.032
        hbg = FancyBboxPatch((hbx, hby), hbw, hbh,
                             boxstyle='round,pad=0.003',
                             facecolor=_COR['cinza_inativo'], edgecolor=_COR['cinza_medio'],
                             linewidth=0.7, alpha=0.88,
                             transform=ax.transAxes, zorder=19, clip_on=False)
        ax.add_patch(hbg); ax.draw_artist(hbg); hbg.remove()
        ht = ax.text(hbx + hbw/2, hby + hbh/2, '?',
                     transform=ax.transAxes, fontsize=7, fontweight='bold',
                     color=_COR['texto_secundario'], ha='center', va='center',
                     zorder=20, clip_on=False)
        ax.draw_artist(ht); ht.remove()

        if self._help_visible:
            ow, oh = 0.25, 0.90
            ox, oy = 0.5 - ow/2, 0.5 - oh/2
            obg = FancyBboxPatch((ox, oy), ow, oh,
                                 boxstyle='round,pad=0.006',
                                 facecolor=_COR['cor_card'], edgecolor=_COR['cinza_medio'],
                                 linewidth=1.0, alpha=0.96,
                                 transform=ax.transAxes, zorder=40, clip_on=False)
            ax.add_patch(obg); ax.draw_artist(obg); obg.remove()
            help_lines = [
                ('LEGEND & GLOSSARY',                             7.5, 'bold',   _COR['texto_principal']),
                ('', 3, 'normal', _COR['texto_principal']),
                ('SIMULATION CARD',                               6,   'bold',   _COR['cinza_medio']),
                ('Stagnation  — gens without ≥1% fitness gain',   6,   'normal', _COR['texto_secundario']),
                ('             Green<5 | Yellow<15 | Red≥15',     6,   'normal', _COR['cinza_medio']),
                ('Avg laps    — mean laps completed/agent last gen', 6, 'normal', _COR['texto_secundario']),
                ('Mutation    — current rate + adaptive mode:',   6,   'normal', _COR['texto_secundario']),
                ('  explore  gen<10, high diversity',             6,   'normal', _COR['cinza_medio']),
                ('  normal   gen<30, balanced',                   6,   'normal', _COR['cinza_medio']),
                ('  fine     gen≥30, converging',                 6,   'normal', _COR['cinza_medio']),
                ('  shake    no gain in 10 gens',                 6,   'normal', _COR['amarelo']),
                ('  plateau  no gain in 20 gens',                 6,   'normal', _COR['vermelho']),
                ('  CRITICAL no gain in 40 gens (max 50% mut)',   6,   'normal', _COR['vermelho']),
                ('', 3, 'normal', _COR['texto_principal']),
                ('EVENTS CARD',                                   6,   'bold',   _COR['cinza_medio']),
                ('Speed     — reward above threshold (20%→90%)',  6,   'normal', _COR['texto_secundario']),
                ('Lap Done  — bonus on each completed lap',       6,   'normal', _COR['texto_secundario']),
                ('Wall      — penalty near wall (post-lap 1)',    6,   'normal', _COR['texto_secundario']),
                ('Wrong way — penalty for going backwards',       6,   'normal', _COR['texto_secundario']),
                ('Too slow  — per-frame penalty, threshold and',  6,   'normal', _COR['texto_secundario']),
                ('           weight escalate each lap',           6,   'normal', _COR['cinza_medio']),
                ('', 3, 'normal', _COR['texto_principal']),
                ('Click ? to close',                              6,   'normal', _COR['cinza_medio']),
            ]
            cy = oy + oh - 0.022
            for htxt, hfs, hfw, hcol in help_lines:
                t = ax.text(ox + 0.014, cy, htxt, transform=ax.transAxes,
                            fontsize=hfs, fontweight=hfw, color=hcol,
                            va='top', ha='left', family='monospace',
                            zorder=41, clip_on=False)
                ax.draw_artist(t); t.remove()
                cy -= (hfs + 1.0) * 0.0058

        # Notificação flutuante de morte do melhor
        NOTIF_TTL = 3.0
        now = time.time()
        self._notificacoes = [(t, m, p) for t, m, p in self._notificacoes if now - t < NOTIF_TTL]
        if self._notificacoes:
            nt, nm, np_ = self._notificacoes[-1]
            alpha = max(0.0, min(1.0, 1.0 - (now - nt) / NOTIF_TTL))
            if alpha > 0:
                nx, ny, nw, nh = 0.870, 0.022, 0.120, 0.048
                nb = FancyBboxPatch((nx, ny), nw, nh,
                                   boxstyle='round,pad=0.004',
                                   facecolor=_COR['cor_card'], edgecolor=_COR['cinza_medio'],
                                   linewidth=0.6, alpha=alpha * 0.85,
                                   transform=ax.transAxes, zorder=19, clip_on=False)
                ax.add_patch(nb); ax.draw_artist(nb); nb.remove()
                lb = FancyBboxPatch((nx, ny), 0.005, nh,
                                   boxstyle='square,pad=0',
                                   facecolor=_COR['vermelho'], edgecolor='none',
                                   alpha=alpha * 0.8,
                                   transform=ax.transAxes, zorder=20, clip_on=False)
                ax.add_patch(lb); ax.draw_artist(lb); lb.remove()
                tm = ax.text(nx + 0.010, ny + nh - 0.020, nm,
                             transform=ax.transAxes, fontsize=7.5, fontweight='bold',
                             color=_COR['texto_principal'], alpha=alpha,
                             verticalalignment='top', zorder=21, clip_on=False)
                ax.draw_artist(tm); tm.remove()
                tp = ax.text(nx + nw - 0.004, ny + nh - 0.020, f'{np_:.0f} pts',
                             transform=ax.transAxes, fontsize=6.5,
                             color=_COR['texto_secundario'], alpha=alpha,
                             verticalalignment='top', horizontalalignment='right',
                             zorder=21, clip_on=False)
                ax.draw_artist(tp); tp.remove()

    def _renderizar_painel_comandos(self, melhor_idx):
        """Painel AI Commands: barras de direção, aceleração e velocidade."""
        melhor_carr = self.carrinhos[melhor_idx] if melhor_idx is not None else self.carrinhos[0]
        acao_m = melhor_carr.ultima_acao
        virar_v = float(acao_m[0])
        acel_v = float(acao_m[1])
        acel_on = acel_v >= 0.5
        vel_pct = melhor_carr.velocidade / CONFIG['carros']['velocidade_max']
        acel_cor = _COR['verde'] if acel_on else _COR['vermelho']
        al = self.ax_log

        # Steering
        lbl_s = al.text(0.5, 8.8, 'STEERING', va='top', ha='left',
                        fontsize=6.5, color=_COR['texto_terciario'], family='monospace', zorder=6)
        al.draw_artist(lbl_s); lbl_s.remove()

        tx0, tx1, ty, th = 0.5, 9.5, 7.6, 0.55
        track = Rectangle((tx0, ty), tx1 - tx0, th,
                           facecolor=_COR['cinza_inativo'], edgecolor='none', zorder=3)
        al.add_patch(track); al.draw_artist(track); track.remove()

        for txt, xp, ha in [('◀', tx0 - 0.15, 'right'), ('▶', tx1 + 0.15, 'left')]:
            lbl = al.text(xp, ty + th/2, txt, va='center', ha=ha,
                          fontsize=5.5, color=_COR['verde'], family='monospace', zorder=6)
            al.draw_artist(lbl); lbl.remove()

        cx = (tx0 + tx1) / 2
        cmark = Rectangle((cx - 0.04, ty), 0.08, th, facecolor=_COR['cinza_medio'], edgecolor='none', zorder=4)
        al.add_patch(cmark); al.draw_artist(cmark); cmark.remove()

        needle_x = cx + virar_v * (tx1 - cx)
        dir_color = (_COR['cor_normal'] if abs(virar_v) < 0.25 else
                     _COR['amarelo']    if virar_v < 0         else _COR['verde'])
        dfill = Rectangle((min(cx, needle_x), ty), abs(needle_x - cx), th,
                           facecolor=dir_color, alpha=0.85, edgecolor='none', zorder=5)
        al.add_patch(dfill); al.draw_artist(dfill); dfill.remove()
        needle = Rectangle((needle_x - 0.08, ty - 0.1), 0.16, th + 0.2,
                            facecolor=_COR['branco'], edgecolor='none', zorder=6)
        al.add_patch(needle); al.draw_artist(needle); needle.remove()

        lbl_sv = al.text(9.5, 8.8, f'{virar_v:+.2f}', va='top', ha='right',
                         fontsize=6.5, color=_COR['texto_secundario'], family='monospace', zorder=6)
        al.draw_artist(lbl_sv); lbl_sv.remove()

        # Throttle
        lbl_a = al.text(0.5, 6.0, 'THROTTLE', va='top', ha='left',
                        fontsize=6.5, color=_COR['texto_terciario'], family='monospace', zorder=6)
        al.draw_artist(lbl_a); lbl_a.remove()
        atrack = Rectangle((tx0, 4.85), tx1 - tx0, th, facecolor=_COR['cinza_inativo'], edgecolor='none', zorder=3)
        al.add_patch(atrack); al.draw_artist(atrack); atrack.remove()
        afill = Rectangle((tx0, 4.85), (tx1 - tx0) * acel_v, th,
                           facecolor=acel_cor, alpha=0.85, edgecolor='none', zorder=4)
        al.add_patch(afill); al.draw_artist(afill); afill.remove()
        lbl_av = al.text(9.5, 6.0, f'{acel_v:.2f}', va='top', ha='right',
                         fontsize=6.5, color=_COR['texto_secundario'], family='monospace', zorder=6)
        al.draw_artist(lbl_av); lbl_av.remove()

        gas_cor = _COR['verde']    if acel_on      else _COR['cinza_medio']
        brake_cor = _COR['vermelho'] if not acel_on  else _COR['cinza_medio']
        for txt, xp, clr in [('GAS', 2.5, gas_cor), ('BRAKE', 7.5, brake_cor)]:
            st = al.text(xp, 3.5, txt, va='center', ha='center', fontsize=7.5,
                         color=clr, family='monospace', fontweight='bold', zorder=6)
            al.draw_artist(st); st.remove()
        sep = al.plot([5.0, 5.0], [3.0, 4.0], color=_COR['cinza_inativo'], linewidth=0.8, zorder=5)
        for l in sep: al.draw_artist(l); l.remove()

        # Speed
        pen = CONFIG['penalidades']
        limiar_spd = (0.0 if melhor_carr.voltas_completas == 0 else
                      min(0.9, pen['limiar_devagar_inicial']
                          + (melhor_carr.voltas_completas - 1) * pen['incremento_limiar_devagar']))
        bar_color = _COR['vermelho'] if vel_pct < limiar_spd else _COR['verde']

        lbl_spd = al.text(0.5, 1.55, 'SPEED', va='top', ha='left',
                          fontsize=6.5, color=_COR['texto_terciario'], family='monospace', zorder=6)
        al.draw_artist(lbl_spd); lbl_spd.remove()
        lbl_spd_v = al.text(9.5, 1.55, f'{vel_pct*100:.0f}%  ({melhor_carr.velocidade:.2f})',
                            va='top', ha='right', fontsize=6.5,
                            color=_COR['texto_secundario'], family='monospace', zorder=6)
        al.draw_artist(lbl_spd_v); lbl_spd_v.remove()
        bg_bar = Rectangle((0.5, 0.3), 9.0, 0.65, color=_COR['cinza_inativo'], zorder=3)
        fill_bar = Rectangle((0.5, 0.3), 9.0 * vel_pct, 0.65, color=bar_color, alpha=0.9, zorder=4)
        for p in (bg_bar, fill_bar): al.add_patch(p); al.draw_artist(p); p.remove()
        if limiar_spd > 0.0:
            marker_x = 0.5 + 9.0 * limiar_spd
            mk = al.plot([marker_x, marker_x], [0.25, 1.05],
                         color=_COR['vermelho'], linewidth=1.2, zorder=5, alpha=0.8)
            for l in mk: al.draw_artist(l); l.remove()

    def _renderizar_painel_eventos(self, melhor_idx):
        """Painel Events: linhas de status de velocidade, volta, parede, contramão e lentidão."""
        ae = self.ax_eventos
        if melhor_idx is None:
            return

        car = self.carrinhos[melhor_idx]
        ef = car.estado_frame
        rec = CONFIG['recompensas']
        pen = CONFIG['penalidades']
        pre_lap = car.voltas_completas == 0

        if pre_lap:
            peso_dev = 0.0
            peso_vel = rec['peso_velocidade']
            limiar   = 0.0
        else:
            v = car.voltas_completas
            peso_dev = pen['peso_devagar_pos_volta'] + (v - 1) * pen['agravante_por_volta']
            peso_vel = rec['peso_velocidade']       + (v - 1) * pen['agravante_por_volta']
            limiar = min(0.9, pen['limiar_devagar_inicial'] + (v - 1) * pen['incremento_limiar_devagar'])

        rows = [
            ('Speed',     'rapido',    _COR['verde']   if not pre_lap else _COR['cinza_medio'],
             '—' if pre_lap else f'+{peso_vel:.1f}/fr',
             'locked pre-lap' if pre_lap else f'above {limiar*100:.0f}% of max speed'),
            ('Lap Done',  'volta',     _COR['verde'],
             f"+{rec['recompensa_volta']:.0f}", 'completed 1 lap'),
            ('Wall',      'parede',    _COR['vermelho'] if not pre_lap else _COR['cinza_medio'],
             '—' if pre_lap else f"−{pen['penalidade_parede_proxima']:.0f}",
             'locked pre-lap' if pre_lap else 'too close to wall'),
            ('Wrong way', 'contramao', _COR['amarelo'] if pre_lap else _COR['vermelho'],
             'FREEZE' if pre_lap else f"−{pen['penalidade_contramao']:.0f}",
             'wrong direction'),
            ('Too slow',  'devagar',   _COR['cinza_medio'] if pre_lap else _COR['amarelo'],
             '—' if pre_lap else f'−{peso_dev:.1f}/fr',
             f'below {limiar*100:.0f}% speed (lap {car.voltas_completas})' if not pre_lap else 'locked pre-lap'),
        ]

        for k, (label, chave, cor_on, val_str, descr) in enumerate(rows):
            gated   = pre_lap and chave in ('rapido', 'parede', 'devagar')
            ativo   = bool(ef.get(chave)) and not gated
            bar_cor = cor_on if ativo else _COR['cinza_inativo']
            lbl_cor = _COR['texto_principal'] if ativo else _COR['texto_terciario']
            val_cor = cor_on if ativo else _COR['cinza_medio']
            dsc_cor = _COR['texto_terciario'] if ativo else _COR['cinza_medio']
            y_row   = 8.4 - k * 1.7

            bar = Rectangle((0.3, y_row - 0.55), 0.25, 1.25,
                             facecolor=bar_cor, edgecolor='none', zorder=4)
            ae.add_patch(bar); ae.draw_artist(bar); bar.remove()

            sep = ae.plot([0.3, 9.7], [y_row - 0.65, y_row - 0.65],
                          color=_COR['cinza_inativo'], linewidth=0.4, zorder=3)
            for l in sep: ae.draw_artist(l); l.remove()

            for xpos, txt, color, ha, fw in [
                (0.9,  label,   lbl_cor, 'left',  'normal'),
                (0.9,  descr,   dsc_cor, 'left',  'normal'),
                (9.7,  val_str, val_cor, 'right', 'bold' if ativo else 'normal'),
            ]:
                ypos = y_row + 0.35 if txt != descr else y_row - 0.15
                fs   = 7.5 if txt != descr else 6.0
                t = ae.text(xpos, ypos, txt, va='center', ha=ha, fontsize=fs,
                            color=color, family='monospace', fontweight=fw, zorder=5)
                ae.draw_artist(t); t.remove()

    def _mostrar_tela_resultado(self, dados, _caminho):
        ax = self.ax_pista
        ax.add_patch(plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor=_COR['preto'], alpha=0.70, zorder=30, clip_on=False,
        ))
        record_str = (f'{dados["record_melhor_volta_frames"]} frames'
                      if dados['record_melhor_volta_frames'] else '---')
        linhas = [
            ('TRAINING COMPLETE',                            20, 'bold',   _COR['verde']),
            ('',                                              6, 'normal', _COR['texto_principal']),
            (f'Generation: {dados["geracao"]}',             14, 'bold',   _COR['texto_principal']),
            (f'Laps completed: {dados["voltas_objetivo"]}',  11, 'normal', _COR['texto_secundario']),
            (f'Best lap: {record_str}',                      11, 'normal', _COR['texto_secundario']),
            (f'Winner car: #{dados["carro_vencedor"]}',      11, 'normal', _COR['texto_secundario']),
            ('',                                              6, 'normal', _COR['texto_principal']),
            ('Results saved to resultados.json',             10, 'normal', _COR['amarelo']),
            ('',                                              5, 'normal', _COR['texto_principal']),
            ('Close the window to exit.',                     9, 'normal', _COR['texto_terciario']),
        ]
        y = 0.70
        for txt, fs, fw, cor in linhas:
            ax.text(0.5, y, txt, transform=ax.transAxes, fontsize=fs,
                    ha='center', va='center', fontweight=fw, color=cor, zorder=31)
            y -= 0.055 + (fs - 9) * 0.004
        self.fig.canvas.draw()

    def _finalizar_treinamento(self):
        self.parar_animacao = True
        print('\n' + '='*70)
        print(f'DONE! Car #{self.vencedor + 1} completed {CONFIG["simulacao"]["voltas_objetivo"]} laps!')
        print(f'Generation: {self.geracao + 1}')
        if self.record_global_volta:
            print(f'Best lap: {self.record_global_volta} frames')
        print('='*70)
        caminho, dados = self._salvar_resultados()
        self._mostrar_tela_resultado(dados, caminho)

    def iniciar_animacao(self):
        print("=" * 60)
        print("AI LEARNING TO DRIVE")
        print("=" * 60)
        print(f"\nGOAL: {CONFIG['simulacao']['voltas_objetivo']} laps without leaving the track")
        print("• GREY  = Track (valid area)")
        print("• GREEN = Off track")
        print("• Press 'Q' or close the window to stop.")
        print("=" * 60)

        try:
            w = self.fig.canvas.manager.window
            w.geometry('1700x1000+40+20')
            w.minsize(1280, 800)
            w.update()
        except Exception:
            try:
                w = self.fig.canvas.manager.window
                w.resize(1700, 1000)
                w.setMinimumSize(1280, 800)
            except Exception:
                pass

        self._preparar_fundo()
        self._bg_refresh_pending = True

        timer = self.fig.canvas.new_timer(interval=0)
        timer.add_callback(self.atualizar_visualizacao)
        timer.start()
        plt.show()
