import numpy as np
from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding


@register_encoding("basis")
class BasisEncoding(Encoding):
    def __init__(self, num_qubits: int, threshold="mid", **kwargs):
        super().__init__(num_qubits, threshold=threshold, **kwargs)
        if isinstance(threshold, str) and threshold.lower() == "mid":
            self.th = float(np.pi)
        else:
            self.th = float(threshold)

    def build(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(f"BasisEncoding: need at least {self.num_qubits} features, got {x.size}.")
        qc = QuantumCircuit(self.num_qubits, name="Basis")
        # Only the first num_qubits features can be mapped to qubits.  The old
        # version iterated over all input features and crashed when a compact
        # qubit count was requested.
        for i, v in enumerate(x[: self.num_qubits]):
            if float(v) > self.th:
                qc.x(i)
        return qc
