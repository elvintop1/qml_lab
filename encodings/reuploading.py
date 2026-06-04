from __future__ import annotations

from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding
from ..utils import edges_for


@register_encoding("reuploading")
@register_encoding("reupload")
@register_encoding("datareuploading")
class ReuploadingEncoding(Encoding):
    """Data re-uploading encoding (Pérez-Salinas et al., 2020 style).

    This corresponds to the "Data Re-uploading" branch in the encoding literature tree.

    Key idea: use a *fixed* (often small) number of qubits, and encode a vector of features
    by applying multiple sequential rotations ("re-uploading" data) across layers.

    This implementation is intentionally general:
      - supports any number of qubits (default: 1)
      - maps features to qubits in a round-robin manner
      - applies rotations per feature and per layer

    Parameters
    ----------
    num_qubits:
        Number of qubits to use (often 1 in the original single-qubit classifier).
    reps:
        Number of re-uploading layers.
    axes:
        Rotation axes pattern, e.g. "ry" | "rz" | "ryrz" | "rxryrz".
        The pattern is applied for *each* feature mapped to a qubit.
    alpha:
        Scale applied to feature values.
    entangle:
        Entanglement pattern between layers: none | ring | full.
        Implemented using CZ gates.
    hadamard:
        If True, apply H on all qubits at the start (often used to start in |+>).
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 2,
        layers: int | None = None,
        axes: str = "ryrz",
        alpha: float = 1.0,
        entangle: str = "none",
        hadamard: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_qubits,
            reps=reps if layers is None else layers,
            layers=layers,
            axes=axes,
            alpha=alpha,
            entangle=entangle,
            hadamard=hadamard,
            **kwargs,
        )
        self.reps = int(reps if layers is None else layers)
        if self.reps <= 0:
            raise ValueError("reps must be >= 1")

        self.axes = str(axes).lower().strip().replace(",", "")
        if any(ch not in "xyzr" for ch in self.axes):
            # very defensive; we expect patterns like 'ryrz' etc.
            pass
        self.alpha = float(alpha)

        self.entangle = str(entangle).lower().strip()
        self.edges: List[Tuple[int, int]] = edges_for(self.num_qubits, self.entangle)
        self.hadamard = bool(hadamard)

    def _apply_rot(self, qc: QuantumCircuit, axis: str, theta: float, q: int):
        axis = axis.lower()
        if axis == "x":
            qc.rx(theta, q)
        elif axis == "y":
            qc.ry(theta, q)
        elif axis == "z":
            qc.rz(theta, q)
        else:
            raise ValueError(f"Unknown rotation axis '{axis}'")

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("ReuploadingEncoding: empty input x")

        n = int(self.num_qubits)
        qc = QuantumCircuit(n, name="Reupload")

        if self.hadamard:
            qc.h(range(n))

        # Pre-parse axis pattern into a list of axes, e.g. 'ryrz' -> ['y','z'].
        # We accept both 'ryrz' and 'yz' style.
        axes: List[str] = []
        s = self.axes
        # Strip any leading 'r'
        s = s.replace("r", "")
        for ch in s:
            if ch in ("x", "y", "z"):
                axes.append(ch)
        if not axes:
            axes = ["y"]

        for _l in range(self.reps):
            # Encode each feature as rotations on a qubit (round-robin)
            for i, v in enumerate(x):
                q = int(i % n)
                theta = self.alpha * float(v)
                for ax in axes:
                    self._apply_rot(qc, ax, theta, q)

            # Optional entanglement between layers
            for (i, j) in self.edges:
                qc.cz(i, j)

        return qc
