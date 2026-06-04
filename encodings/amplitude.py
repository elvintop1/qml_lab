import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from .base import Encoding
from .registry import register_encoding
@register_encoding("amplitude")
class AmplitudeEncoding(Encoding):
    def __init__(self, num_qubits: int, rescale: str = "unit", epsilon: float = 1e-12, **kwargs):
        super().__init__(num_qubits, rescale=rescale, epsilon=epsilon, **kwargs)
        self.rescale = str(rescale).lower()
        self.eps = float(epsilon)
    def _prep(self, x):
        v = np.asarray(x, dtype=float)
        if self.rescale == "unit":
            v = v / (2.0 * np.pi)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < self.eps:
            v = np.zeros_like(v); v[0] = 1.0
        else:
            v = v / n
        return v
    def build(self, x):
        d = len(x); N = 2 ** self.num_qubits
        v = self._prep(x)
        amp = np.zeros(N, dtype=complex); amp[:min(d, N)] = v[:min(d, N)]
        n = np.linalg.norm(amp)
        if not np.isfinite(n) or n < self.eps:
            amp = np.zeros(N, dtype=complex); amp[0] = 1.0
        else:
            amp = amp / n
        qc = QuantumCircuit(self.num_qubits, name="Amp")
        qc.append(StatePreparation(amp), range(self.num_qubits))
        return qc
