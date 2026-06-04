from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding


def _quantize_levels(x: np.ndarray, levels: int, *, eps: float = 1e-9) -> np.ndarray:
    """Quantize scaled features x in [0, 2π] into integers [0, levels-1]."""
    x = np.asarray(x, dtype=float).ravel()
    # Map to [0,1)
    x01 = x / (2.0 * np.pi)
    x01 = np.clip(x01, 0.0, 1.0 - eps)
    q = np.floor(x01 * int(levels)).astype(np.int64)
    q = np.clip(q, 0, int(levels) - 1)
    return q


def _rolling_hash(values: np.ndarray, mod: int, seed: int = 0, prime: int = 1315423911) -> int:
    """A fast deterministic rolling hash to map an int vector -> [0, mod)."""
    h = int(seed) & 0xFFFFFFFF
    m = int(mod)
    if m <= 0:
        return 0
    for v in np.asarray(values, dtype=np.int64).ravel():
        # 32-bit mix
        h ^= (int(v) + prime + ((h << 6) & 0xFFFFFFFF) + (h >> 2)) & 0xFFFFFFFF
    return int(h % m)


@register_encoding("integer")
@register_encoding("binaryinteger")
@register_encoding("binary_integer")
class IntegerBasisEncoding(Encoding):
    """Binary / integer basis encoding.

    This corresponds to the "Binary / Integer" branch in the encoding literature tree.

    In many applications, "integer" basis encoding means directly encoding an integer
    value as a computational basis state using its binary representation.

    For general tabular vectors (continuous), we need a deterministic mapping from x -> integer.
    We implement:
      1) quantize each feature into `levels` bins
      2) hash the quantized vector into an integer in [0, 2**n_bits)
      3) set qubits with X gates for bits that are 1 (little-endian)

    Parameters
    ----------
    num_qubits:
        Number of qubits == n_bits.
    levels:
        Quantization levels per feature.
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
        n_bits = int(self.num_qubits)
        mod = 2 ** n_bits
        q = _quantize_levels(x, self.levels)
        code = _rolling_hash(q, mod=mod, seed=self.seed)

        qc = QuantumCircuit(n_bits, name="Integer")
        for b in range(n_bits):
            if (code >> b) & 1:
                qc.x(b)
        return qc
