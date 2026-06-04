import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate

from .base import Encoding
from .registry import register_encoding
from ..utils import edges_for


@register_encoding("zzprod")
@register_encoding("zz_product")
class ZZProductEncoding(Encoding):
    """Product ZZ feature map.

    Older code only applied single-qubit RZ rotations and therefore did not
    contain any ZZ product interaction despite the encoding name.  This version
    uses duplicate-free entangling edges and RZZ phases proportional to x_i x_j.
    """

    def __init__(
        self,
        num_qubits: int,
        scale: float = 1.0,
        *,
        single_scale: float | None = None,
        pair_scale: float | None = None,
        entangler: str = "ring",
        hadamard: bool = True,
        final_hadamard: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_qubits,
            scale=scale,
            single_scale=single_scale,
            pair_scale=pair_scale,
            entangler=entangler,
            hadamard=hadamard,
            final_hadamard=final_hadamard,
            **kwargs,
        )
        self.scale = float(scale)
        self.single_scale = float(scale if single_scale is None else single_scale)
        self.pair_scale = float(scale if pair_scale is None else pair_scale)
        self.edges = edges_for(num_qubits, entangler)
        self.hadamard = bool(hadamard)
        self.final_hadamard = bool(final_hadamard)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(f"ZZProductEncoding: need at least {self.num_qubits} features, got {x.size}.")

        qc = QuantumCircuit(self.num_qubits, name="ZZProd")
        if self.hadamard:
            qc.h(range(self.num_qubits))
        for i in range(self.num_qubits):
            qc.rz(-2.0 * self.single_scale * float(x[i]), i)
        for i, j in self.edges:
            qc.append(RZZGate(-2.0 * self.pair_scale * float(x[i] * x[j])), [i, j])
        if self.final_hadamard:
            qc.h(range(self.num_qubits))
        return qc
