from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding
from .integer import _quantize_levels, _rolling_hash


@register_encoding("onehot")
@register_encoding("one_hot")
class OneHotBasisEncoding(Encoding):
    """One-hot basis encoding.

    This corresponds to the "One-Hot" branch in the encoding literature tree.

    For general continuous vectors, we map x -> category id using the same quantize+hash
    strategy as IntegerBasisEncoding, then encode the category as a unary basis state:
      |0...010...0>

    Parameters
    ----------
    num_qubits:
        Number of categories (qubits). The active category is indicated by an X gate.
    levels:
        Quantization levels per feature before hashing.
    seed:
        Hash seed.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        levels: int = 16,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(num_qubits, levels=levels, seed=seed, **kwargs)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError("levels must be >= 2")
        self.seed = int(seed)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        n_cat = int(self.num_qubits)
        if n_cat <= 0:
            raise ValueError("OneHotBasisEncoding: num_qubits must be > 0")

        q = _quantize_levels(x, self.levels)
        cat = _rolling_hash(q, mod=n_cat, seed=self.seed)

        qc = QuantumCircuit(n_cat, name="OneHot")
        qc.x(int(cat))
        return qc
