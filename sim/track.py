"""
Track geometry, SDF lookup, car physics and simulation entities.
No UI/rendering code here — pure simulation logic.
"""

import sys
import numpy as np
import json
import os
from collections import deque


# ---------------------------------------------------------------------------
# RESOURCE PATH — works both in dev and in a PyInstaller frozen build
# ---------------------------------------------------------------------------

def _resource_path(relative_path: str) -> str:
    """Return absolute path to a bundled resource.

    When the app is frozen by PyInstaller (--onefile or --onedir),
    ``sys._MEIPASS`` points to the extraction directory where data files
    are placed.  In a normal Python environment the project root is used.
    """
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

def carregar_config():
    """Carrega configurações do arquivo config.json"""
    config_path = _resource_path('config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


CONFIG = carregar_config()
_COR = CONFIG['cores_ui']


# ---------------------------------------------------------------------------
# GEOMETRIA DO CIRCUITO — lida exclusivamente de pista.json
# ---------------------------------------------------------------------------

_pista_json_path = _resource_path('pista.json')
with open(_pista_json_path, 'r', encoding='utf-8') as _f:
    _pj = json.load(_f)

_CTRL_CIRCUITO = np.array(_pj['pontos_controle'])
_NOME_PISTA    = _pj.get('nome', 'Track')
CONFIG.setdefault('pista', {})['largura_pista'] = _pj['largura_pista']

_larg = _pj['largada']
CONFIG.setdefault('linha_largada', {}).update({
    'x':          _larg['x'],
    'y':          _larg['y'],
    'angulo':     _larg['angulo_linha_graus'],
    'largura':    _larg.get('largura_linha', _pj['largura_pista']),
    'altura':     0.4,
    'num_zebras': 14,
})
CONFIG['carros']['angulo_inicial'] = _larg['angulo_carro_graus']

_cheg = _pj['chegada']
CONFIG.setdefault('linha_chegada', {}).update({
    'x':          _cheg['x'],
    'y':          _cheg['y'],
    'angulo':     _cheg['angulo_linha_graus'],
    'largura':    _cheg.get('largura_linha', _pj['largura_pista']),
    'altura':     0.4,
    'num_zebras': 14,
})

print(f'[pista.json] {len(_CTRL_CIRCUITO)} control points | width {_pj["largura_pista"]:.3f}')


def _catmull_rom(pts, n_seg=40):
    """Gera poligonal suavizada (Catmull-Rom) atraves dos pontos de controle."""
    n = len(pts) - 1   # pts[0] == pts[-1] — circuito fechado
    ts = np.linspace(0, 1, n_seg, endpoint=False)
    tt = ts * ts;  ttt = tt * ts
    result = []
    for i in range(n):
        p0 = pts[i - 1] if i > 0 else pts[n - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 <= n else pts[(i + 2) - n]
        seg = 0.5 * (
            np.outer(-ttt + 2*tt - ts,    p0) +
            np.outer( 3*ttt - 5*tt + 2,   p1) +
            np.outer(-3*ttt + 4*tt + ts,   p2) +
            np.outer( ttt - tt,            p3)
        )
        result.append(seg)
    return np.vstack(result)   # (n * n_seg, 2)


# Centerline pre-computada
_CENTERLINE = _catmull_rom(_CTRL_CIRCUITO)   # (N, 2)
_CL_X = np.append(_CENTERLINE[:, 0], _CENTERLINE[0, 0])
_CL_Y = np.append(_CENTERLINE[:, 1], _CENTERLINE[0, 1])

# Viewport: bounding box + margem
_PAD_X = 6.0
_PAD_Y = 3.0
_TRACK_XLIM = (float(_CL_X.min() - _PAD_X), float(_CL_X.max() + _PAD_X))
_TRACK_YLIM = (float(_CL_Y.min() - _PAD_Y), float(_CL_Y.max() + _PAD_Y))


# ---------------------------------------------------------------------------
# SDF (Signed Distance Field) — grade pré-computada O(1) para inside-track
# ---------------------------------------------------------------------------

def _build_sdf(res: float = 0.08) -> tuple:
    pad = 3.0
    xmin = float(_CL_X.min()) - pad
    xmax = float(_CL_X.max()) + pad
    ymin = float(_CL_Y.min()) - pad
    ymax = float(_CL_Y.max()) + pad

    xs = np.arange(xmin, xmax, res, dtype=np.float32)
    ys = np.arange(ymin, ymax, res, dtype=np.float32)
    GX, GY = np.meshgrid(xs, ys)

    ax_ = _CL_X[:-1].astype(np.float32)
    ay_ = _CL_Y[:-1].astype(np.float32)
    bx_ = _CL_X[1:].astype(np.float32)
    by_ = _CL_Y[1:].astype(np.float32)
    ddx = bx_ - ax_;  ddy = by_ - ay_
    denom = np.maximum(ddx*ddx + ddy*ddy, np.float32(1e-12))

    min_d2 = np.full(GX.shape, np.inf, dtype=np.float32)
    for i in range(len(ax_)):
        t  = np.clip(((GX - ax_[i])*ddx[i] + (GY - ay_[i])*ddy[i]) / denom[i],
                     np.float32(0.0), np.float32(1.0))
        d2 = (ax_[i] + t*ddx[i] - GX)**2 + (ay_[i] + t*ddy[i] - GY)**2
        np.minimum(min_d2, d2, out=min_d2)

    return min_d2, xmin, ymin, float(res)


print('[SDF] Pre-computing track distance grid...', end=' ', flush=True)
_SDF_GRID, _SDF_XMIN, _SDF_YMIN, _SDF_RES = _build_sdf()
print(f'OK ({_SDF_GRID.shape[1]}×{_SDF_GRID.shape[0]} cells)')


def ponto_dentro_pista(x: float, y: float) -> bool:
    """O(1): consulta a grade SDF pré-computada."""
    hw2 = (CONFIG['pista']['largura_pista'] / 2.0) ** 2
    xi  = int((x - _SDF_XMIN) / _SDF_RES)
    yi  = int((y - _SDF_YMIN) / _SDF_RES)
    if xi < 0 or yi < 0 or xi >= _SDF_GRID.shape[1] or yi >= _SDF_GRID.shape[0]:
        return False
    return float(_SDF_GRID[yi, xi]) <= hw2


def _multiponto_dentro_pista(xs_arr: np.ndarray, ys_arr: np.ndarray) -> np.ndarray:
    """Versão vetorizada: consulta SDF para arrays de pontos. Retorna bool array."""
    hw2   = np.float32((CONFIG['pista']['largura_pista'] / 2.0) ** 2)
    xi    = ((xs_arr - _SDF_XMIN) / _SDF_RES).astype(np.int32)
    yi    = ((ys_arr - _SDF_YMIN) / _SDF_RES).astype(np.int32)
    H, W  = _SDF_GRID.shape
    valid = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
    d2    = np.full(len(xs_arr), np.inf, dtype=np.float32)
    d2[valid] = _SDF_GRID[yi[valid], xi[valid]]
    return d2 <= hw2


def calcular_tangente_pista(x, y, _pista_config=None):
    """Retorna vetor tangente na direcao de marcha da pista no ponto (x,y)."""
    ax_ = _CL_X[:-1];  ay_ = _CL_Y[:-1]
    bx_ = _CL_X[1:];   by_ = _CL_Y[1:]
    dx = bx_ - ax_;    dy = by_ - ay_
    denom = dx*dx + dy*dy
    denom = np.where(denom > 1e-12, denom, 1.0)
    t  = np.clip(((x - ax_)*dx + (y - ay_)*dy) / denom, 0.0, 1.0)
    d2 = (ax_ + t*dx - x)**2 + (ay_ + t*dy - y)**2
    idx = int(np.argmin(d2))
    mag = np.sqrt(dx[idx]**2 + dy[idx]**2)
    if mag < 1e-8:
        return np.array([1.0, 0.0])
    return np.array([dx[idx] / mag, dy[idx] / mag])


def gerar_contornos_pista():
    """Gera contornos externo e interno da pista por offset da centerline."""
    hw = CONFIG['pista']['largura_pista'] / 2.0
    n  = len(_CENTERLINE)
    tx = np.zeros(n);  ty = np.zeros(n)
    for i in range(n):
        d   = _CENTERLINE[(i + 1) % n] - _CENTERLINE[(i - 1) % n]
        mag = np.sqrt(d[0]**2 + d[1]**2)
        if mag > 1e-12:
            tx[i], ty[i] = d[0] / mag, d[1] / mag
        else:
            tx[i], ty[i] = 1.0, 0.0
    ext_x = _CENTERLINE[:, 0] + hw * ty
    ext_y = _CENTERLINE[:, 1] - hw * tx
    int_x = _CENTERLINE[:, 0] - hw * ty
    int_y = _CENTERLINE[:, 1] + hw * tx
    return ext_x, ext_y, int_x, int_y


def calcular_posicao_inicial_carro(indice, total_carros):
    """Calcula posição inicial do carro dentro da linha de largada."""
    linha        = CONFIG['linha_largada']
    x_centro     = linha['x']
    y_centro     = linha['y']
    largura_util = linha['largura'] * 0.8

    if total_carros == 1:
        offset_relativo = 0
    else:
        offset_relativo = (indice / (total_carros - 1) - 0.5)

    x_local = offset_relativo * largura_util
    y_local = 0.0

    rotacao_rad  = np.radians(linha['angulo'])
    x_rotacionado = x_local * np.cos(rotacao_rad) - y_local * np.sin(rotacao_rad)
    y_rotacionado = x_local * np.sin(rotacao_rad) + y_local * np.cos(rotacao_rad)

    return x_centro + x_rotacionado, y_centro + y_rotacionado, CONFIG['carros']['angulo_inicial']


def cruza_linha_chegada(x0, y0, x1, y1):
    """Retorna True se o segmento (x0,y0)→(x1,y1) cruza a linha de chegada."""
    linha        = CONFIG['linha_chegada']
    cx, cy       = linha['x'], linha['y']
    rot          = -np.radians(linha['angulo'])
    cos_r, sin_r = np.cos(rot), np.sin(rot)

    def _rot(px, py):
        rx, ry = px - cx, py - cy
        return rx * cos_r - ry * sin_r, rx * sin_r + ry * cos_r

    x0r, y0r = _rot(x0, y0)
    x1r, y1r = _rot(x1, y1)

    if y0r * y1r > 0 or y0r == y1r:
        return False

    t      = y0r / (y0r - y1r)
    x_cross = x0r + t * (x1r - x0r)
    return bool(abs(x_cross) <= linha['largura'] / 2)


# ---------------------------------------------------------------------------
# ENTIDADE: Carrinho controlado por IA
# ---------------------------------------------------------------------------

class CarrinhoIA:
    """Carrinho controlado por rede neural."""

    def __init__(self, indice=0, total_carros=1):
        x, y, angulo      = calcular_posicao_inicial_carro(indice, total_carros)
        self.x             = x
        self.y             = y
        self.velocidade    = 0.0
        self.angulo        = angulo
        self.vivo          = True
        self.tempo_vivo    = 0
        self.trajetoria_x  = []
        self.trajetoria_y  = []
        self.voltas_completas  = 0
        self.frame_inicio_volta = 0
        self.melhor_tempo_volta = None
        self.velocidade_total   = 0.0
        self.pontos_acumulados  = 0.0
        self.indice_carro       = indice
        self.total_carros       = total_carros
        self.motivo_morte       = None

    def reset(self):
        """Recomeça a tentativa."""
        x, y, angulo          = calcular_posicao_inicial_carro(self.indice_carro, self.total_carros)
        self.x                 = x
        self.y                 = y
        self.velocidade        = (CONFIG['carros']['velocidade_max']
                                  * CONFIG['carros']['velocidade_inicial_pct'] / 100.0)
        self.angulo            = angulo
        self.vivo              = True
        self.tempo_vivo        = 0
        self.trajetoria_x      = [self.x]
        self.trajetoria_y      = [self.y]
        self.voltas_completas  = 0
        self.frame_inicio_volta = 0
        self.melhor_tempo_volta = None
        self.velocidade_total   = 0.0
        self.pontos_acumulados  = 0.0
        self.max_afastamento_chegada   = 0.0
        self.ultima_acao       = np.array([0.0, 0.5])
        self._pts_janela       = deque(maxlen=CONFIG['penalidades']['janela_perda_continua'])
        self._pts_frame_inicio = 0.0
        self.cl_checkpoint     = 0
        self.frames_checkpoint = int(np.random.randint(
            0, CONFIG['simulacao']['frames_sem_progresso']))
        self.max_cl_index          = 0
        self.cl_index_inicio_volta = 0
        self.eventos               = deque(maxlen=10)
        self.estado_frame          = {}
        self.motivo_morte          = None

    def get_sensores(self, pista_config=None):
        """8 sensores: -90,-45,-22.5,0,+22.5,+45,+90 graus + velocidade.
        Totalmente vetorizado via SDF."""
        _OFFSETS = np.array([-90., -45., -22.5, 0., 22.5, 45., 90.], dtype=np.float32)
        rads  = np.radians(self.angulo + _OFFSETS)
        dists = np.arange(0.2, 15.0, 0.2, dtype=np.float32)

        sx = np.float32(self.x) + np.cos(rads)[:, None] * dists
        sy = np.float32(self.y) + np.sin(rads)[:, None] * dists

        dentro = _multiponto_dentro_pista(sx.ravel(), sy.ravel()).reshape(7, 74)

        result  = np.full(7, 15.0, dtype=np.float32)
        fora    = ~dentro
        tem_hit = fora.any(axis=1)
        if tem_hit.any():
            idx_hit         = np.argmax(fora, axis=1)
            result[tem_hit] = dists[idx_hit[tem_hit]]

        vel_norm = self.velocidade / CONFIG['carros']['velocidade_max']
        return np.append(result / 15.0, vel_norm)

    def mover(self, acao):
        """Move o carrinho baseado na ação da rede neural."""
        if not self.vivo:
            return

        prev_x, prev_y = self.x, self.y

        self.angulo += acao[0] * CONFIG['carros']['angulo_virada_max']
        vmax  = CONFIG['carros']['velocidade_max']
        passo = CONFIG['carros']['passo_velocidade']

        gas = float(acao[1])
        if gas >= 0.5:
            self.velocidade = min(vmax, self.velocidade + passo)
        else:
            self.velocidade = max(0.0, self.velocidade - passo)

        rad    = np.radians(self.angulo)
        self.x += self.velocidade * np.cos(rad)
        self.y += self.velocidade * np.sin(rad)

        # Detecta cruzamento da linha de chegada
        if cruza_linha_chegada(prev_x, prev_y, self.x, self.y):
            _progresso_minimo = len(_CENTERLINE) * CONFIG['deteccao_volta']['fracao_minima_volta']
            _progresso_real   = self.max_cl_index - self.cl_index_inicio_volta
            if (self.max_afastamento_chegada >= CONFIG['deteccao_volta']['afastamento_minimo_da_chegada']
                    and _progresso_real >= _progresso_minimo):
                tempo_desta_volta = self.tempo_vivo - self.frame_inicio_volta
                eh_recorde        = (self.melhor_tempo_volta is None
                                     or tempo_desta_volta < self.melhor_tempo_volta)
                melhoria_frames   = ((self.melhor_tempo_volta - tempo_desta_volta)
                                     if (eh_recorde and self.melhor_tempo_volta is not None) else 0)
                if eh_recorde:
                    self.melhor_tempo_volta = tempo_desta_volta

                self.frame_inicio_volta        = self.tempo_vivo
                self.voltas_completas         += 1
                self.max_afastamento_chegada   = 0.0
                self.cl_index_inicio_volta     = self.max_cl_index
                self.cl_checkpoint             = self.max_cl_index
                self.frames_checkpoint         = 0

                rec = CONFIG['recompensas']
                self.pontos_acumulados += rec['recompensa_volta']
                self.eventos.append((f"+{rec['recompensa_volta']:.0f} LAP!", _COR['verde']))
                self.estado_frame['volta'] = True

                target    = rec['target_frames_volta']
                bonus_vel = rec['bonus_volta_rapida'] * target / max(tempo_desta_volta, 1)
                self.pontos_acumulados += bonus_vel
                self.eventos.append((f'+{bonus_vel:.0f} spd ({tempo_desta_volta}fr)', _COR['verde']))

                if melhoria_frames > 0:
                    antigo_tempo = tempo_desta_volta + melhoria_frames
                    melhoria_pct = melhoria_frames / antigo_tempo
                    bonus_rec    = (rec['recompensa_volta']
                                    * melhoria_pct
                                    * self.voltas_completas
                                    * rec['fator_bonus_recorde'])
                    self.pontos_acumulados += bonus_rec
                    self.eventos.append((f'+{bonus_rec:.0f} RECORD!', _COR['amarelo']))

        self.trajetoria_x.append(self.x)
        self.trajetoria_y.append(self.y)
        self.velocidade_total += self.velocidade
        self.tempo_vivo       += 1

        # Atualiza max_afastamento_chegada
        lc           = CONFIG['linha_chegada']
        dist_chegada = np.sqrt((self.x - lc['x'])**2 + (self.y - lc['y'])**2)
        if dist_chegada > self.max_afastamento_chegada:
            self.max_afastamento_chegada = dist_chegada

        # Sistema de velocidade com limiar dinâmico
        rec  = CONFIG['recompensas']
        pen  = CONFIG['penalidades']
        vmax = CONFIG['carros']['velocidade_max']
        vel_norm = self.velocidade / vmax

        if self.voltas_completas == 0:
            pts_vel    = 0.0
            limiar_dev = 0.0
        else:
            peso_dev   = pen['peso_devagar_pos_volta'] + (self.voltas_completas - 1) * pen['agravante_por_volta']
            limiar_dev = min(0.9, pen['limiar_devagar_inicial']
                            + (self.voltas_completas - 1) * pen['incremento_limiar_devagar'])
            peso_vel   = rec['peso_velocidade'] + (self.voltas_completas - 1) * pen['agravante_por_volta']
            if vel_norm >= limiar_dev:
                pts_vel = (vel_norm - limiar_dev) / (1.0 - limiar_dev) * peso_vel
            else:
                pts_vel = -((limiar_dev - vel_norm) / limiar_dev) * peso_dev

        self.pontos_acumulados += pts_vel

        if self.voltas_completas > 0 and vel_norm < limiar_dev:
            self.estado_frame['devagar'] = True
        else:
            self.estado_frame['rapido'] = True

        # Kill por perda contínua
        self._pts_janela.append(self.pontos_acumulados)
        if (self.tempo_vivo > self._pts_janela.maxlen
                and len(self._pts_janela) == self._pts_janela.maxlen):
            perda_janela = self._pts_janela[-1] - self._pts_janela[0]
            limiar_perda = (CONFIG['penalidades']['perda_continua_limiar_pre_volta']
                            if self.voltas_completas == 0
                            else CONFIG['penalidades']['perda_continua_limiar_pos_volta'])
            score_minimo = CONFIG['penalidades']['score_minimo_kill']
            if perda_janela < limiar_perda and self.pontos_acumulados < score_minimo:
                self.vivo         = False
                self.motivo_morte = 'Continuous point loss'
                self.eventos.append(('-mort Cont. loss', _COR['vermelho']))

    def checar_colisao(self, pista_config=None):
        """Verifica se saiu da pista."""
        if not ponto_dentro_pista(self.x, self.y):
            self.vivo              = False
            self.motivo_morte      = 'Off track'
            self.pontos_acumulados -= CONFIG['penalidades']['penalidade_colisao']
            self.eventos.append(('CRASH!', _COR['vermelho']))
            return True
        return False
