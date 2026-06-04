from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from .base import Encoding
from .registry import register_encoding


@register_encoding("histogram")
@register_encoding("probability")
@register_encoding("probabilityhistogram")
class HistogramAmplitudeEncoding(Encoding):
    """Probability / histogram amplitude encoding.

    This corresponds to the "Probability / Histogram" branch in the thesis
    encoding taxonomy.

    In the literature, specialized loaders (e.g., Grover–Rudolph) can exploit
    structure of probability distributions to prepare states more efficiently.

    For this benchmark repo we implement a *robust* variant that:
      1) converts a real-valued vector into a non-negative probability vector p
      2) loads sqrt(p) as amplitudes using Qiskit's StatePreparation

    This is suitable for comparing *classical preparation overhead* at scale
    (scaling + circuit construction) across encoding families.

    Parameters
    ----------
    num_qubits:
        Number of qubits. The input dimension is padded/truncated to 2**num_qubits.
    mode:
        How to convert input x into non-negative weights:
          - 'abs'     : w_i = |x_i|
          - 'relu'    : w_i = max(x_i, 0)
          - 'softmax' : w = exp(x - max(x))
    epsilon:
        Numerical stabilizer for normalization.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        mode: str = "abs",
        epsilon: float = 1e-12,
        **kwargs,
    ):
        super().__init__(num_qubits, mode=mode, epsilon=epsilon, **kwargs)
        self.mode = str(mode).lower().strip()
        if self.mode not in {"abs", "relu", "softmax"}:
            raise ValueError("mode must be one of: abs, relu, softmax")
        self.eps = float(epsilon)

    def _weights(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if self.mode == "abs":
            w = np.abs(x)
        elif self.mode == "relu":
            w = np.maximum(x, 0.0)
        else:
            # softmax weights
            z = x - float(np.max(x))
            w = np.exp(z)
        s = float(np.sum(w))
        if not np.isfinite(s) or s < self.eps:
            # fallback to a trivial distribution
            w = np.zeros_like(x)
            if w.size:
                w[0] = 1.0
            return w
        return w / s

    def build(self, x: np.ndarray) -> QuantumCircuit:
        p = self._weights(x)
        # amplitude vector is sqrt(prob)
        v = np.sqrt(np.maximum(p, 0.0))

        N = 2 ** int(self.num_qubits)
        amp = np.zeros(N, dtype=complex)
        d = int(v.size)
        amp[: min(d, N)] = v[: min(d, N)]

        n = float(np.linalg.norm(amp))
        if not np.isfinite(n) or n < self.eps:
            amp = np.zeros(N, dtype=complex)
            amp[0] = 1.0
        else:
            amp = amp / n

        qc = QuantumCircuit(self.num_qubits, name="HistAmp")
        qc.append(StatePreparation(amp), range(self.num_qubits))
        return qc
