from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate

from .base import Encoding
from .registry import register_encoding
from ..utils import edges_for
from .pairfuncs import pair_phi as pair_phi_factory


@register_encoding("havlicek")
@register_encoding("iqp")
@register_encoding("zz_feature_map")
class HavlicekEncoding(Encoding):
    """IQP / Havlíček-style feature map.

    The older default used ``entangler='none'``.  That made the encoding almost
    an angle map and was the main reason the IQP baseline looked much weaker
    than expected.  The default is now a duplicate-free ring, which keeps the
    circuit shallow while actually using ZZ correlations.
    """

    def __init__(
        self,
        num_qubits: int,
        entangler: str = "ring",
        pair: str = "prod",
        reps: int = 1,
        **kwargs,
    ):
        super().__init__(num_qubits, entangler=entangler, pair=pair, reps=reps, **kwargs)
        self.edges: List[Tuple[int, int]] = edges_for(num_qubits, entangler)
        self.single_phi = lambda x, i: float(x[i])
        self.pair_phi = pair_phi_factory(pair)
        self.reps = max(1, int(reps))

    def _apply_Udiag(self, qc: QuantumCircuit, x: np.ndarray):
        for i in range(self.num_qubits):
            qc.rz(-2.0 * self.single_phi(x, i), i)
        for (i, j) in self.edges:
            qc.append(RZZGate(-2.0 * self.pair_phi(x, i, j)), [i, j])

    def build(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(f"HavlicekEncoding: need at least {self.num_qubits} features, got {x.size}.")
        qc = QuantumCircuit(self.num_qubits, name="HavlicekIQP")
        for _ in range(self.reps):
            qc.h(range(self.num_qubits))
            self._apply_Udiag(qc, x)
            qc.h(range(self.num_qubits))
            self._apply_Udiag(qc, x)
        return qc
