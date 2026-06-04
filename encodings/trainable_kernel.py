from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from qiskit import QuantumCircuit

from .base import Encoding
from .registry import register_encoding
from ..utils import edges_for


@register_encoding("trainablekernel")
@register_encoding("trainable_kernel")
@register_encoding("trainablekernelmap")
class TrainableKernelEncoding(Encoding):
    """Trainable kernel (parametric) feature map.

    This corresponds to the "Trainable Kernel" branch in the encoding literature tree.

    In the literature, trainable kernels typically learn parameters to maximize
    kernel alignment or classification performance.

    For this thesis benchmark repo, we provide a *fixed-parameter* trainable feature map:
      - the circuit depends on x (data) and on a set of fixed parameters w
      - w is initialized randomly (seed-controlled) and kept constant during runs

    This makes it possible to include the "trainable kernel" family in the
    encoding comparison without adding another optimization loop.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    reps:
        Number of layers.
    entangler:
        none | ring | full (CZ entanglers).
    seed:
        Random seed for generating fixed weights.
    weight_scale:
        Uniform range for weights: U(-weight_scale, weight_scale).
    data_scale:
        Scale applied to data rotations.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        reps: int = 2,
        layers: int | None = None,
        entangler: str = "ring",
        seed: int = 0,
        weight_scale: float = 0.3,
        data_scale: float = 1.0,
        hadamard: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_qubits,
            reps=reps if layers is None else layers,
            layers=layers,
            entangler=entangler,
            seed=seed,
            weight_scale=weight_scale,
            data_scale=data_scale,
            hadamard=hadamard,
            **kwargs,
        )

        self.reps = int(reps if layers is None else layers)
        if self.reps <= 0:
            raise ValueError("reps must be >= 1")

        self.entangler = str(entangler).lower().strip()
        self.edges: List[Tuple[int, int]] = edges_for(num_qubits, self.entangler)

        self.seed = int(seed)
        self.weight_scale = float(weight_scale)
        self.data_scale = float(data_scale)
        self.hadamard = bool(hadamard)

        # Pre-generate fixed weights: per layer per qubit (Ry, Rz).
        rng = np.random.default_rng(self.seed)
        self.w_ry = rng.uniform(-self.weight_scale, self.weight_scale, size=(self.reps, num_qubits)).astype(float)
        self.w_rz = rng.uniform(-self.weight_scale, self.weight_scale, size=(self.reps, num_qubits)).astype(float)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.num_qubits:
            raise ValueError(
                f"TrainableKernelEncoding: need at least {self.num_qubits} features, got {x.size}."
            )

        qc = QuantumCircuit(self.num_qubits, name="TrainKernel")

        if self.hadamard:
            qc.h(range(self.num_qubits))

        for l in range(self.reps):
            for q in range(self.num_qubits):
                # Data-dependent part
                theta = self.data_scale * float(x[q])
                qc.ry(theta, q)
                qc.rz(theta, q)
                # Fixed ("trainable") weights
                qc.ry(float(self.w_ry[l, q]), q)
                qc.rz(float(self.w_rz[l, q]), q)

            for (i, j) in self.edges:
                qc.cz(i, j)

        return qc
