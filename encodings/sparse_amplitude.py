from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from .base import Encoding
from .registry import register_encoding


@register_encoding("sparse")
@register_encoding("sparseamplitude")
@register_encoding("sparse_amplitude")
class SparseAmplitudeEncoding(Encoding):
    """Sparse-vector amplitude encoding.

    This corresponds to the "Sparse Vector" branch in the thesis encoding taxonomy.

    In principle, sparse state preparation can be more efficient than general
    arbitrary state preparation when most amplitudes are zero.

    In this benchmark implementation we approximate "sparse" preparation by:
      - thresholding small entries to zero (classical preprocessing)
      - using Qiskit's generic StatePreparation to build a circuit

    This keeps the code stable across Qiskit versions and still allows
    meaningful measurement of *classical circuit construction overhead*.

    Parameters
    ----------
    num_qubits:
        Number of qubits. Vector is padded/truncated to 2**num_qubits.
    rescale:
        If 'unit', divide inputs by 2π before normalization (keeps angles in [0,1]
        when inputs were scaled to [0,2π]).
    threshold:
        Values with |v_i| < threshold are set to 0.
    epsilon:
        Numerical stabilizer.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        rescale: str = "unit",
        threshold: float = 1e-3,
        epsilon: float = 1e-12,
        **kwargs,
    ):
        super().__init__(num_qubits, rescale=rescale, threshold=threshold, epsilon=epsilon, **kwargs)
        self.rescale = str(rescale).lower().strip()
        self.th = float(threshold)
        self.eps = float(epsilon)

    def _prep(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=float).ravel()
        if self.rescale == "unit":
            v = v / (2.0 * np.pi)
        # sparsify
        v = np.where(np.abs(v) < self.th, 0.0, v)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < self.eps:
            v = np.zeros_like(v)
            if v.size:
                v[0] = 1.0
            return v
        return v / n

    def build(self, x: np.ndarray) -> QuantumCircuit:
        v = self._prep(x)
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

        qc = QuantumCircuit(self.num_qubits, name="SparseAmp")
        qc.append(StatePreparation(amp), range(self.num_qubits))
        return qc
