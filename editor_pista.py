"""
Interactive Track Editor
========================
Click to add control points and generate the smoothed circuit.
Export to pista.json to use in the simulator.

Mouse controls:
  Left click       — add point / drag existing point
  Right click      — remove nearest point
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import json
import os
import tkinter as tk  # apenas para clipboard (Ctrl+C/V)

def _carregar_config_editor():
    cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(cp, 'r', encoding='utf-8') as f:
        return json.load(f)

_E_COR = _carregar_config_editor()['cores_ui']

# ---------------------------------------------------------------------------
# Catmull-Rom (fechado): pts deve ser array de pontos únicos; o script
# acrescenta pts[0] no final para fechar o loop.
# ---------------------------------------------------------------------------

def _catmull_rom(pts, n_seg=40):
    """Suaviza lista de pontos únicos como spline de Catmull-Rom fechado."""
    closed = np.vstack([pts, pts[0]])   # fecha o circuito
    n = len(closed) - 1
    ts  = np.linspace(0, 1, n_seg, endpoint=False)
    tt  = ts * ts
    ttt = tt * ts
    result = []
    for i in range(n):
        p0 = closed[i - 1] if i > 0 else closed[n - 1]
        p1 = closed[i]
        p2 = closed[i + 1]
        p3 = closed[i + 2] if i + 2 <= n else closed[(i + 2) - n]
        seg = 0.5 * (
            np.outer(-ttt + 2*tt - ts,   p0) +
            np.outer( 3*ttt - 5*tt + 2,  p1) +
            np.outer(-3*ttt + 4*tt + ts,  p2) +
            np.outer( ttt - tt,           p3)
        )
        result.append(seg)
    return np.vstack(result)

def _gerar_bordas(centerline, hw):
    """Gera offset ±hw a partir da centerline fechada."""
    n = len(centerline)
    tx = np.zeros(n); ty = np.zeros(n)
    cl_c = np.vstack([centerline, centerline[0]])
    for i in range(n):
        d   = cl_c[(i + 1)] - cl_c[(i - 1) % n]
        mag = np.hypot(d[0], d[1])
        tx[i], ty[i] = (d / mag) if mag > 1e-12 else (1.0, 0.0)
    ext_x = centerline[:, 0] + hw * ty
    ext_y = centerline[:, 1] - hw * tx
    int_x = centerline[:, 0] - hw * ty
    int_y = centerline[:, 1] + hw * tx
    return ext_x, ext_y, int_x, int_y

def _angulo_seg(p1, p2):
    """Ângulo em graus do segmento p1→p2."""
    d = p2 - p1
    return float(np.degrees(np.arctan2(d[1], d[0])))

# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 50, 35   # espaco de coordenadas disponivel para desenhar
SNAP_RADIUS = 0.8              # raio para remover ponto com clique direito

PISTA_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pista.json')

class EditorPista:
    def __init__(self):
        self.pontos = []          # lista de [x, y] (pontos únicos de controle)
        self.largura = 1.8
        self.nome_pista = 'Track 1'
        self._arrastar_idx = None

        # Carrega ultima pista salva, se existir
        self._carregar_pista_json()

        # --- Figura ---
        self.fig = plt.figure(figsize=(19, 10))
        self.fig.patch.set_facecolor(_E_COR['editor_fundo'])

        # Área principal — ocupa quase tudo, barra inferior compacta
        _bar_h = 0.13
        self.ax = self.fig.add_axes([0.03, _bar_h + 0.01, 0.94, 0.86 - 0.01])
        self._config_canvas()

        # --- Separador ---
        ax_sep = self.fig.add_axes([0.0, _bar_h - 0.002, 1.0, 0.002])
        ax_sep.set_facecolor(_E_COR['cinza_inativo'])
        ax_sep.set_axis_off()

        # Margens verticais dentro da barra
        _ey = 0.025          # y base dos elementos
        _eh = 0.075          # altura dos elementos

        # --- Rótulos das seções ---
        _lbl_y = _ey + _eh + 0.004
        self.fig.text(0.03,  _lbl_y, 'TRACK NAME', fontsize=6.5,
                      fontweight='bold', color=_E_COR['texto_secundario'],
                      va='bottom', family='monospace')
        self.fig.text(0.285, _lbl_y, 'ACTIONS', fontsize=6.5,
                      fontweight='bold', color=_E_COR['texto_secundario'],
                      va='bottom', family='monospace')
        self.fig.text(0.58,  _lbl_y, 'TRACK WIDTH', fontsize=6.5,
                      fontweight='bold', color=_E_COR['texto_secundario'],
                      va='bottom', family='monospace')

        # --- TextBox: nome da pista (compacto) ---
        ax_nome = self.fig.add_axes([0.03, _ey, 0.22, _eh])
        self.txt_nome = TextBox(ax_nome, '',
                                initial=self.nome_pista,
                                color=_E_COR['cor_asfalto'],
                                hovercolor=_E_COR['cinza_inativo'])
        self.txt_nome.text_disp.set_color(_E_COR['branco'])
        self.txt_nome.text_disp.set_fontsize(9)
        self.txt_nome.text_disp.set_fontfamily('monospace')
        self.txt_nome.text_disp.set_fontweight('bold')
        for sp in ax_nome.spines.values():
            sp.set_edgecolor(_E_COR['cinza_medio'])
            sp.set_linewidth(1.2)
        self.txt_nome.on_text_change(self._cb_nome_change)

        # --- Botões (compactos) ---
        _bw = 0.075
        _gap = 0.008
        _bx = 0.285
        ax_save  = self.fig.add_axes([_bx,              _ey, _bw, _eh])
        ax_undo  = self.fig.add_axes([_bx + _bw + _gap, _ey, _bw, _eh])
        ax_clear = self.fig.add_axes([_bx + 2*(_bw + _gap), _ey, _bw, _eh])

        self.btn_salvar = Button(ax_save,  'SAVE',  color='#2d5c2e', hovercolor=_E_COR['verde'])
        self.btn_undo   = Button(ax_undo,  'UNDO',  color='#2a3a4a', hovercolor=_E_COR['cinza_medio'])
        self.btn_limpar = Button(ax_clear, 'CLEAR', color='#5c2d2d', hovercolor=_E_COR['vermelho'])

        for btn, lbl_cor in [(self.btn_salvar, _E_COR['verde']),
                              (self.btn_undo,   _E_COR['branco']),
                              (self.btn_limpar, _E_COR['vermelho'])]:
            btn.label.set_fontsize(9)
            btn.label.set_fontweight('bold')
            btn.label.set_color(lbl_cor)
            btn.label.set_family('monospace')
            for sp in btn.ax.spines.values():
                sp.set_edgecolor(_E_COR['cinza_medio'])
                sp.set_linewidth(1.0)

        # --- Slider de largura ---
        ax_slider = self.fig.add_axes([0.58, _ey + 0.018, 0.37, 0.035])
        self.slider_larg = Slider(ax_slider, '', 0.5, 5.0,
                                   valinit=self.largura, color=_E_COR['verde'])
        self.slider_larg.poly.set_alpha(0.7)
        self.slider_larg.valtext.set_color(_E_COR['branco'])
        self.slider_larg.valtext.set_fontsize(9)
        self.slider_larg.valtext.set_fontfamily('monospace')
        ax_slider.set_facecolor(_E_COR['cinza_inativo'])
        for sp in ax_slider.spines.values():
            sp.set_edgecolor(_E_COR['cinza_medio'])

        # --- Callbacks ---
        self.btn_salvar.on_clicked(self._cb_salvar)
        self.btn_undo.on_clicked(self._cb_undo)
        self.btn_limpar.on_clicked(self._cb_limpar)
        self.slider_larg.on_changed(self._cb_largura)
        self.fig.canvas.mpl_connect('key_press_event',      self._cb_key_clipboard)
        self.fig.canvas.mpl_connect('button_press_event',   self._cb_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._cb_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._cb_mouse_release)

        self._atualizar()
        plt.show()

    def _cb_key_clipboard(self, event):
        """Suporte a Ctrl+C / Ctrl+V no TextBox."""
        if not self.txt_nome.active:
            return
        try:
            root = tk.Tk(); root.withdraw()
            if event.key == 'ctrl+c':
                root.clipboard_clear()
                root.clipboard_append(self.txt_nome.text)
                root.update()
            elif event.key == 'ctrl+v':
                pasted = root.clipboard_get()
                cur = self.txt_nome.text
                self.txt_nome.set_val(cur + pasted)
            root.destroy()
        except Exception:
            pass

    def _cb_nome_change(self, text):
        self.nome_pista = text.strip() if text.strip() else 'Track 1'

    # ------------------------------------------------------------------
    # Callbacks de widgets
    def _carregar_pista_json(self):
        """Carrega pista.json se existir, restaurando pontos e largura."""
        if not os.path.exists(PISTA_JSON_PATH):
            return
        try:
            with open(PISTA_JSON_PATH, 'r', encoding='utf-8') as f:
                pj = json.load(f)
            pts = pj.get('pontos_controle', [])
            if len(pts) >= 3:
                # Remove ponto duplicado de fechamento se existir
                arr = np.array(pts)
                if np.allclose(arr[0], arr[-1]):
                    arr = arr[:-1]
                self.pontos = arr.tolist()
            self.largura = float(pj.get('largura_pista', 1.8))
            self.nome_pista = pj.get('nome', 'Track 1')
            print(f'[editor] Track loaded: {len(self.pontos)} points, width {self.largura:.2f}')
        except Exception as e:
            print(f'[editor] Warning: could not load pista.json: {e}')

    # ------------------------------------------------------------------
    def _config_canvas(self):
        ax = self.ax
        ax.set_xlim(-1, CANVAS_W + 1)
        ax.set_ylim(-1, CANVAS_H + 1)
        ax.set_facecolor(_E_COR['editor_grama'])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color=_E_COR['branco'], alpha=0.08, linewidth=0.7)
        ax.tick_params(colors=_E_COR['editor_tick'])
        ax.set_title('Track Editor  —  left click: add / drag   •   right click: remove point',
                     color=_E_COR['branco'], fontsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_E_COR['texto_secundario'])

    # ------------------------------------------------------------------
    # Callbacks de widgets
    def _cb_largura(self, val):
        self.largura = float(val)
        self._atualizar()

    def _cb_undo(self, _=None):
        if self.pontos:
            self.pontos.pop()
            self._atualizar()

    def _cb_limpar(self, _=None):
        self.pontos.clear()
        self._atualizar()

    # ------------------------------------------------------------------
    # Mouse
    def _cb_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # esquerdo — inicia arrasto ou adiciona
            idx = self._ponto_proximo(x, y)
            if idx is not None:
                self._arrastar_idx = idx   # vai arrastar
            else:
                self.pontos.append([x, y])
                self._atualizar()

        elif event.button == 3:  # direito — remove ponto próximo
            idx = self._ponto_proximo(x, y)
            if idx is not None:
                self.pontos.pop(idx)
                self._atualizar()

    def _cb_mouse_move(self, event):
        if self._arrastar_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.pontos[self._arrastar_idx] = [event.xdata, event.ydata]
        self._atualizar()

    def _cb_mouse_release(self, event):
        self._arrastar_idx = None

    def _ponto_proximo(self, x, y):
        """Retorna índice do ponto de controle mais próximo ou None."""
        if not self.pontos:
            return None
        pts = np.array(self.pontos)
        d2 = (pts[:, 0] - x)**2 + (pts[:, 1] - y)**2
        idx = int(np.argmin(d2))
        if np.sqrt(d2[idx]) <= SNAP_RADIUS:
            return idx
        return None

    # ------------------------------------------------------------------
    def _atualizar(self):
        ax = self.ax
        ax.clear()
        self._config_canvas()

        n = len(self.pontos)

        # Ajusta viewport ao bounding box dos pontos (com margem), se existirem
        if n >= 1:
            pts_np = np.array(self.pontos)
            pad = max(self.largura * 3, 2.0)
            ax.set_xlim(pts_np[:, 0].min() - pad, pts_np[:, 0].max() + pad)
            ax.set_ylim(pts_np[:, 1].min() - pad, pts_np[:, 1].max() + pad)

        if n < 3:
            # Mostra apenas pontos e mensagem
            if self.pontos:
                pts = np.array(self.pontos)
                ax.plot(pts[:, 0], pts[:, 1], 'o--',
                        color=_E_COR['cor_ponto_controle'], markersize=9, linewidth=1.5, alpha=0.7)
                for i, (px, py) in enumerate(self.pontos, 1):
                    ax.text(px + 0.2, py + 0.2, str(i),
                            color=_E_COR['cor_ponto_controle'], fontsize=8, fontweight='bold')
            msg = f'Add at least 3 points  ({n}/3)'
            cx = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
            cy = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            ax.text(cx, cy, msg,
                    ha='center', va='center', fontsize=13,
                    color=_E_COR['branco'], alpha=0.5)
            self.fig.canvas.draw_idle()
            return

        # Centerline suavizada
        pts_arr = np.array(self.pontos)
        cl = _catmull_rom(pts_arr)
        hw = self.largura / 2.0
        ext_x, ext_y, int_x, int_y = _gerar_bordas(cl, hw)

        # Preenche asfalto — compound path (anel sem seam)
        N, M = len(ext_x), len(int_x)
        outer_v = np.column_stack([ext_x, ext_y])
        inner_v = np.column_stack([int_x[::-1], int_y[::-1]])
        verts = np.concatenate([outer_v, outer_v[0:1], inner_v, inner_v[0:1]])
        codes = np.array(
            [MplPath.MOVETO] + [MplPath.LINETO] * (N - 1) + [MplPath.CLOSEPOLY] +
            [MplPath.MOVETO] + [MplPath.LINETO] * (M - 1) + [MplPath.CLOSEPOLY],
            dtype=np.uint8)
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=_E_COR['cor_asfalto'], edgecolor='none', zorder=1, alpha=0.90))

        # Bordas brancas
        ax.plot(np.append(ext_x, ext_x[0]), np.append(ext_y, ext_y[0]),
                color=_E_COR['branco'], linewidth=2.5, zorder=3)
        ax.plot(np.append(int_x, int_x[0]), np.append(int_y, int_y[0]),
                color=_E_COR['branco'], linewidth=2.5, zorder=3)

        # Centerline tracejada
        ax.plot(np.append(cl[:, 0], cl[0, 0]),
                np.append(cl[:, 1], cl[0, 1]),
                '--', color=_E_COR['amarelo'], linewidth=1, alpha=0.45, zorder=4)

        # Linha de largada/chegada — zebra alinhada com a BORDA real da pista
        # Encontra ponto da centerline mais próximo do mid pt0→pt1 e usa sua tangente
        mid = (pts_arr[0] + pts_arr[1]) / 2.0
        dists_cl = np.hypot(cl[:, 0] - mid[0], cl[:, 1] - mid[1])
        idx_cl   = int(np.argmin(dists_cl))
        # Tangente na centerline (usando pontos vizinhos)
        prev_cl  = cl[(idx_cl - 1) % len(cl)]
        next_cl  = cl[(idx_cl + 1) % len(cl)]
        tang_cl  = next_cl - prev_cl
        ang_carro = float(np.degrees(np.arctan2(tang_cl[1], tang_cl[0])))
        ang_linha = ang_carro + 90.0
        cl_mid    = cl[idx_cl]   # ponto exato na centerline
        n_zebras  = 14
        w_zebra   = (2 * hw) / n_zebras
        zebra_h   = w_zebra * 1.2
        x_start   = cl_mid[0] - hw
        y_start   = cl_mid[1] - zebra_h / 2
        for i in range(n_zebras):
            cor = _E_COR['branco'] if i % 2 == 0 else _E_COR['preto']
            t = Affine2D().rotate_deg_around(cl_mid[0], cl_mid[1], ang_linha) + ax.transData
            r = Rectangle((x_start + i * w_zebra, y_start), w_zebra, zebra_h,
                           facecolor=cor, edgecolor='none', zorder=6)
            r.set_transform(t)
            ax.add_patch(r)

        # Seta de direção (na centerline também)
        tang_norm = tang_cl / np.hypot(tang_cl[0], tang_cl[1]) if np.hypot(tang_cl[0], tang_cl[1]) > 1e-4 else tang_cl
        ax.annotate('', xy=cl_mid + tang_norm * 0.9 * hw,
                    xytext=cl_mid,
                    arrowprops=dict(arrowstyle='->', color=_E_COR['vermelho'],
                                    lw=2.0, mutation_scale=14),
                    zorder=8)

        # Pontos de controle
        pts_np = np.array(self.pontos)
        ax.scatter(pts_np[:, 0], pts_np[:, 1],
                   s=80, color=_E_COR['cor_ponto_controle'], edgecolors=_E_COR['preto'],
                   linewidths=1.2, zorder=9)
        for i, (px, py) in enumerate(self.pontos, 1):
            ax.text(px + 0.2, py + 0.3, str(i),
                    color=_E_COR['cor_ponto_controle'], fontsize=8, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground=_E_COR['preto'])])

        # Info
        info = f'{n} points  |  width: {self.largura:.1f}  |  start line fixed between pt1–pt2  |  dir: {ang_carro:.0f}°'
        ax.text(0.01, 0.01, info,
                transform=ax.transAxes, fontsize=9, color=_E_COR['branco'], alpha=0.7,
                bbox=dict(boxstyle='round', facecolor=_E_COR['preto'], alpha=0.3))

        # Nome da pista — topo esquerdo
        ax.text(0.012, 0.978, self.nome_pista.upper(),
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=_E_COR['branco'], va='top', ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=_E_COR['preto'],
                          edgecolor=_E_COR['btn_slider'], alpha=0.7), zorder=10)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _cb_salvar(self, _=None):
        n = len(self.pontos)
        if n < 3:
            print('Need at least 3 points to save.')
            return

        pts_arr = np.array(self.pontos)
        cl  = _catmull_rom(pts_arr)
        hw  = self.largura / 2.0

        # Garante que o array exportado fecha o circuito (último == primeiro)
        pontos_fechados = pts_arr.tolist()
        pontos_fechados.append(pontos_fechados[0])

        # Tangente real da centerline no ponto mais próximo do mid pt0→pt1
        mid_raw  = (pts_arr[0] + pts_arr[1]) / 2.0
        dists_cl = np.hypot(cl[:, 0] - mid_raw[0], cl[:, 1] - mid_raw[1])
        idx_cl   = int(np.argmin(dists_cl))
        prev_cl  = cl[(idx_cl - 1) % len(cl)]
        next_cl  = cl[(idx_cl + 1) % len(cl)]
        tang_cl  = next_cl - prev_cl
        mid      = cl[idx_cl]
        ang_carro = float(np.degrees(np.arctan2(tang_cl[1], tang_cl[0])))
        ang_linha  = ang_carro + 90.0

        pista_json = {
            'nome': self.nome_pista,
            'pontos_controle': pontos_fechados,
            'largura_pista': float(self.largura),
            'largada': {
                'x': float(mid[0]),
                'y': float(mid[1]),
                'largura_linha': float(self.largura),
                'angulo_carro_graus': float(ang_carro),
                'angulo_linha_graus': float(ang_linha),
            },
            'chegada': {
                'x': float(mid[0]),
                'y': float(mid[1]),
                'largura_linha': float(self.largura),
                'angulo_linha_graus': float(ang_linha),
            },
        }

        caminho = PISTA_JSON_PATH
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(pista_json, f, indent=2)

        print(f'Track saved to: {caminho}')
        print(f'  {n} control points  |  width {self.largura:.1f}  |  initial dir {ang_carro:.0f}°')


if __name__ == '__main__':
    print(__doc__)
    EditorPista()
