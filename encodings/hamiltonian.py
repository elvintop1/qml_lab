from __future__ import annotations

from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate

from .base import Encoding
from .pairfuncs import pair_phi as pair_phi_factory
from .registry import register_encoding
from ..utils import edges_for


@register_encoding("hamiltonian")
@register_encoding("hamiltonianevolution")
@register_encoding("hamiltonian_evolution")
class HamiltonianEvolutionEncoding(Encoding):
    """Hamiltonian / correlation encoding via (commuting) time evolution.

    This corresponds to the "Hamiltonian Evolution" branch in the encoding
    literature tree.

    We implement a simple diagonal Hamiltonian of the form:
      H(x) = sum_i  phi_i(x) Z_i  +  sum_(i,j in edges)  phi_{ij}(x) Z_i Z_j

    Since all terms commute, time evolution can be implemented exactly using
    RZ and RZZ rotations:
      exp(-i t * a Z)   == RZ(2 t a)
      exp(-i t * b ZZ)  == RZZ(2 t b)

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    entangler:
        Which interaction graph to use for ZZ terms: none | ring | full.
    pair:
        Pairwise coefficient function name (see pairfuncs.py): prod | cos | gauss | lin.
    time:
        Evolution time t.
    hadamard:
        If True, apply H on all qubits before evolution (start in |+> basis).
        If False, evolve from |0>.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        entangler: str = "ring",
        pair: str = "lin",
        time: float = 1.0,
        hadamard: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_qubits,
            entangler=entangler,
            pair=pair,
            time=time,
            hadamard=hadamard,
            **kwargs,
        )
        self.edges: List[Tuple[int, int]] = edges_for(num_qubits, str(entangler))
        self.pair_phi = pair_phi_factory(pair)
        self.time = float(time)
        self.hadamard = bool(hadamard)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(
                f"HamiltonianEvolutionEncoding: need at least {self.num_qubits} features, got {x.size}."
            )

        qc = QuantumCircuit(self.num_qubits, name="HamEvol")

        if self.hadamard:
            qc.h(range(self.num_qubits))

        t = self.time

        # Single-qubit Z terms
        for i in range(self.num_qubits):
            a = float(x[i])
            qc.rz(2.0 * t * a, i)

        # ZZ interaction terms
        for (i, j) in self.edges:
            b = float(self.pair_phi(x, i, j))
            qc.append(RZZGate(2.0 * t * b), [i, j])

        return qc
