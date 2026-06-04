import numpy as np
from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding


@register_encoding("angle")
class AngleEncoding(Encoding):
    """
    Angle encoding: apply a single rotation per qubit.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    axis:
        'rx' | 'ry' | 'rz'
    alpha:
        Scaling factor applied to each feature value (after the caller's scaling).
    """
    def __init__(self, num_qubits: int, axis: str = "ry", alpha: float = 1.0, **kwargs):
        super().__init__(num_qubits, axis=axis, alpha=alpha, **kwargs)
        axis = str(axis).lower().strip()
        if axis not in ("rx", "ry", "rz"):
            raise ValueError("axis must be one of: 'rx', 'ry', 'rz'")
        self.axis = axis
        self.alpha = float(alpha)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(f"AngleEncoding: need at least {self.num_qubits} features, got {x.size}.")
        qc = QuantumCircuit(self.num_qubits, name="Angle")
        for i in range(self.num_qubits):
            theta = self.alpha * float(x[i])
            if self.axis == "rx":
                qc.rx(theta, i)
            elif self.axis == "ry":
                qc.ry(theta, i)
            else:
                qc.rz(theta, i)
        return qc
