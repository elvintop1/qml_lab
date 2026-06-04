import numpy as np
from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding
from ..utils import edges_for


@register_encoding("denseangle")
class DenseAngleEncoding(Encoding):
    """
    Dense angle encoding with two classical features per qubit by default.

    If the feature count is odd, the final missing ``phi`` angle is padded with
    zero.  This is deliberate: rejecting 13-feature tabular datasets makes the
    benchmark unfair even though the circuit can represent the partial pair.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        scale_theta: float = 1.0,
        scale_phi: float = 1.0,
        entangle: str = "ring",   # "none" | "linear" | "ring" | "full"
        features_per_qubit: int = 2,
        **kwargs,
    ):
        super().__init__(
            num_qubits,
            scale_theta=scale_theta,
            scale_phi=scale_phi,
            entangle=entangle,
            features_per_qubit=features_per_qubit,
            **kwargs,
        )
        self.scale_theta = float(scale_theta)
        self.scale_phi = float(scale_phi)
        self.entangle = str(entangle).lower().strip()
        # Validate with the shared helper so 2-qubit rings do not duplicate CZ.
        self.edges = edges_for(self.num_qubits, self.entangle)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(
                f"DenseAngleEncoding: need at least {self.num_qubits} theta features, got {x.size}."
            )

        qc = QuantumCircuit(self.num_qubits, name="DenseAngle")

        for q in range(self.num_qubits):
            theta = float(x[2 * q]) if (2 * q) < x.size else 0.0
            phi = float(x[2 * q + 1]) if (2 * q + 1) < x.size else 0.0
            qc.ry(self.scale_theta * theta, q)
            qc.rz(self.scale_phi * phi, q)

        for i, j in self.edges:
            qc.cz(i, j)

        return qc
