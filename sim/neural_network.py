"""
Neural Network that controls the cars.
Architecture: 8 inputs (7 distance sensors + speed) → 14 hidden → 2 outputs (steer, throttle)
"""

import numpy as np


class RedeNeuralCarrinho:
    """Rede Neural que controla o carrinho — 8 entradas (7 sensores + velocidade), 14 ocultos, 2 saidas"""

    def __init__(self):
        self.W1 = np.random.randn(8, 14) * 0.5
        self.b1 = np.zeros(14)
        self.W2 = np.random.randn(14, 2) * 0.5
        self.b2 = np.zeros(2)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def decidir_acao(self, sensores):
        """Forward pass para um único carro."""
        a1 = self._sigmoid(sensores @ self.W1 + self.b1)
        a2 = self._sigmoid(a1 @ self.W2 + self.b2)
        return np.array([(a2[0] - 0.5) * 2.0, a2[1]])

    @staticmethod
    def forward_batch(redes, sensor_matrix):
        """Forward pass vetorizado para N carros de uma vez.

        redes         — lista de RedeNeuralCarrinho (tamanho N)
        sensor_matrix — np.ndarray (N, 8)
        Retorna acoes — np.ndarray (N, 2)  [virar, acelerar]
        """
        N = len(redes)
        W1 = np.stack([r.W1 for r in redes])   # (N, 8, 14)
        b1 = np.stack([r.b1 for r in redes])   # (N, 14)
        W2 = np.stack([r.W2 for r in redes])   # (N, 14, 2)
        b2 = np.stack([r.b2 for r in redes])   # (N, 2)

        z1 = np.einsum('ni,nij->nj', sensor_matrix, W1) + b1
        a1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -500, 500)))
        z2 = np.einsum('ni,nij->nj', a1, W2) + b2
        a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))

        acoes = np.empty((N, 2), dtype=np.float64)
        acoes[:, 0] = (a2[:, 0] - 0.5) * 2.0   # virar  (-1..+1)
        acoes[:, 1] = a2[:, 1]                   # acelerar (0..1)
        return acoes

    def mutar(self, taxa=0.1):
        self.W1 += np.random.randn(*self.W1.shape) * taxa
        self.W2 += np.random.randn(*self.W2.shape) * taxa
        self.b1 += np.random.randn(*self.b1.shape) * taxa
        self.b2 += np.random.randn(*self.b2.shape) * taxa

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
