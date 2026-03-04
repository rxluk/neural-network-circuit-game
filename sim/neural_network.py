import numpy as np


class RedeNeuralCarrinho:
    # Arquitetura: 8 entradas → 14 neurônios ocultos → 2 saídas (virar, acelerar)

    def __init__(self):
        self.W1 = np.random.randn(8, 14) * 0.5
        self.b1 = np.zeros(14)
        self.W2 = np.random.randn(14, 2) * 0.5
        self.b2 = np.zeros(2)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def decidir_acao(self, sensores):
        a1 = self._sigmoid(sensores @ self.W1 + self.b1)
        a2 = self._sigmoid(a1 @ self.W2 + self.b2)
        return np.array([(a2[0] - 0.5) * 2.0, a2[1]])

    @staticmethod
    def forward_batch(redes, sensor_matrix):
        # Forward pass vetorizado para N carros simultaneamente.
        W1 = np.stack([r.W1 for r in redes])
        b1 = np.stack([r.b1 for r in redes])
        W2 = np.stack([r.W2 for r in redes])
        b2 = np.stack([r.b2 for r in redes])

        a1 = 1.0 / (1.0 + np.exp(-np.clip(np.einsum('ni,nij->nj', sensor_matrix, W1) + b1, -500, 500)))
        a2 = 1.0 / (1.0 + np.exp(-np.clip(np.einsum('ni,nij->nj', a1, W2) + b2, -500, 500)))

        acoes = np.empty((len(redes), 2), dtype=np.float64)
        acoes[:, 0] = (a2[:, 0] - 0.5) * 2.0  # virar:    -1 (esquerda) a +1 (direita)
        acoes[:, 1] = a2[:, 1]                  # acelerar:  0 (freio)   a  1 (gás)
        return acoes

    def mutar(self, taxa=0.1):
        for attr in ('W1', 'b1', 'W2', 'b2'):
            mat = getattr(self, attr)
            setattr(self, attr, mat + np.random.randn(*mat.shape) * taxa)

    def crossover(self, outro):
        for attr in ('W1', 'b1', 'W2', 'b2'):
            a = getattr(self, attr)
            b = getattr(outro, attr)
            mask = np.random.rand(*a.shape) > 0.5
            setattr(self, attr, np.where(mask, a, b))

    def copiar_de(self, outra_rede):
        self.W1 = outra_rede.W1.copy()
        self.b1 = outra_rede.b1.copy()
        self.W2 = outra_rede.W2.copy()
        self.b2 = outra_rede.b2.copy()
