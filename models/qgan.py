from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def _int_to_bits(i: int, n_bits: int) -> List[int]:
    return [(i >> b) & 1 for b in range(n_bits)]  # little-endian


def _batch_ints_to_bits(ints: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.zeros((len(ints), n_bits), dtype=np.float32)
    for r, v in enumerate(ints):
        for b in range(n_bits):
            out[r, b] = float((int(v) >> b) & 1)
    return out


def _all_bitstrings(n_bits: int) -> np.ndarray:
    n = 2 ** n_bits
    return _batch_ints_to_bits(np.arange(n, dtype=np.int64), n_bits)


def _discretize_1d(x: np.ndarray, n_qubits: int, transform: str = "log1p") -> Tuple[np.ndarray, Dict]:
    """
    Map 1D real-valued samples to integer bins in [0, 2**n_qubits).

    Returns
    -------
    idx: np.ndarray[int64]
    meta: dict with inverse-transform info
    """
    x = np.asarray(x, dtype=float).ravel()
    if transform not in ("none", "log1p"):
        raise ValueError("transform must be 'none' or 'log1p'")

    if transform == "log1p":
        x_t = np.log1p(np.maximum(x, 0.0))
    else:
        x_t = x

    xmin = float(np.min(x_t))
    xmax = float(np.max(x_t))
    if xmax <= xmin:
        xmax = xmin + 1.0

    x01 = (x_t - xmin) / (xmax - xmin)
    x01 = np.clip(x01, 0.0, 0.999999)

    L = 2 ** int(n_qubits)
    idx = np.floor(x01 * L).astype(np.int64)
    idx = np.clip(idx, 0, L - 1)

    meta = {"transform": transform, "xmin": xmin, "xmax": xmax, "n_qubits": int(n_qubits)}
    return idx, meta


def _undiscretize_1d(idx: np.ndarray, meta: Dict) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    n_qubits = int(meta["n_qubits"])
    L = 2 ** n_qubits
    x01 = (idx.astype(float) + 0.5) / L
    xmin = float(meta["xmin"])
    xmax = float(meta["xmax"])
    x_t = x01 * (xmax - xmin) + xmin

    transform = meta.get("transform", "none")
    if transform == "log1p":
        x = np.expm1(x_t)
    else:
        x = x_t
    return x


class QuantumGenerator:
    """
    Quantum generator circuit producing a probability distribution over 2**n_qubits bitstrings.
    Implemented as a parameterized hardware-efficient circuit.
    """
    def __init__(self, n_qubits: int, layers: int = 2):
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)

        # params: per layer per qubit: Ry + Rz
        self.num_params = self.layers * self.n_qubits * 2
        self.theta = ParameterVector("theta", self.num_params)
        self.template = self._build_template()

    def _build_template(self) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n, name="QGAN_G")
        t = list(self.theta)

        idx = 0
        for _l in range(self.layers):
            for q in range(n):
                qc.ry(t[idx], q); idx += 1
                qc.rz(t[idx], q); idx += 1
            if n >= 2:
                for q in range(n - 1):
                    qc.cz(q, q + 1)
                qc.cz(n - 1, 0)
        return qc

    def probs(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float).ravel()
        if params.size != self.num_params:
            raise ValueError(f"Expected {self.num_params} generator params, got {params.size}")
        bind = {p: float(v) for p, v in zip(self.theta, params)}
        try:
            qc = self.template.assign_parameters(bind, inplace=False)
        except Exception:
            qc = self.template.bind_parameters(bind)  # older qiskit fallback
        sv = Statevector.from_instruction(qc).data
        return (np.abs(sv) ** 2).astype(float)


class Discriminator(nn.Module):
    def __init__(self, n_bits: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class QGANResult:
    generator_params: np.ndarray
    history: Dict[str, List[float]]
    discretization: Dict
    timings: Dict[str, float]


def train_qgan_1d(
    real_values: np.ndarray,
    *,
    n_qubits: int = 6,
    gen_layers: int = 2,
    disc_hidden: int = 32,
    batch_size: int = 256,
    epochs: int = 200,
    d_steps: int = 1,
    g_steps: int = 1,
    lr_d: float = 1e-3,
    lr_g: float = 5e-2,
    transform: str = "log1p",
    seed: int = 0,
    verbose: bool = True,
) -> QGANResult:
    """
    Train a simple QGAN to learn a 1D distribution.

    - Real data is discretized into 2**n_qubits bins.
    - Generator is a PQC producing a categorical distribution over bitstrings.
    - Discriminator is a small MLP operating on bitstrings.

    This is intended for research / experimentation (not production-grade).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx_real, meta = _discretize_1d(real_values, n_qubits=n_qubits, transform=transform)
    n_states = 2 ** int(n_qubits)

    G = QuantumGenerator(n_qubits=n_qubits, layers=gen_layers)
    D = Discriminator(n_bits=n_qubits, hidden=disc_hidden)

    optD = optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCELoss()

    # init generator params
    theta = rng.uniform(-0.1, 0.1, size=(G.num_params,)).astype(float)

    all_bits = torch.tensor(_all_bitstrings(n_qubits), dtype=torch.float32)

    history = {"loss_D": [], "loss_G": []}
    timings: Dict[str, float] = {}

    t_start = time.perf_counter()

    for ep in range(1, int(epochs) + 1):
        # ---- Train discriminator ----
        for _ in range(int(d_steps)):
            # sample real
            real_batch = rng.choice(idx_real, size=min(batch_size, len(idx_real)), replace=True)
            x_real = torch.tensor(_batch_ints_to_bits(real_batch, n_qubits), dtype=torch.float32)
            y_real = torch.ones((x_real.shape[0], 1), dtype=torch.float32)

            # sample fake from generator distribution
            p = G.probs(theta)
            fake_batch = rng.choice(np.arange(n_states), size=x_real.shape[0], replace=True, p=p)
            x_fake = torch.tensor(_batch_ints_to_bits(fake_batch, n_qubits), dtype=torch.float32)
            y_fake = torch.zeros((x_fake.shape[0], 1), dtype=torch.float32)

            # discriminator loss
            optD.zero_grad(set_to_none=True)
            out_real = D(x_real)
            out_fake = D(x_fake)
            lossD = 0.5 * (bce(out_real, y_real) + bce(out_fake, y_fake))
            lossD.backward()
            optD.step()

        # ---- Train generator ----
        for _ in range(int(g_steps)):
            # Evaluate discriminator on all possible bitstrings (exact expectation)
            with torch.no_grad():
                d_all = D(all_bits).squeeze(1)  # shape (2**n,)
                # non-saturating GAN uses log D(fake)
                v = torch.log(d_all + 1e-9).cpu().numpy().astype(float)  # values per state

            # Generator objective: maximize E_{z~p_theta}[log D(z)]
            # L_G = -E[log D] => gradient is -dE/dtheta
            shift = math.pi / 2.0
            grad = np.zeros_like(theta)

            for i in range(theta.size):
                tp = theta.copy(); tp[i] += shift
                tm = theta.copy(); tm[i] -= shift
                p_plus = G.probs(tp)
                p_minus = G.probs(tm)
                E_plus = float(np.dot(p_plus, v))
                E_minus = float(np.dot(p_minus, v))
                dE = 0.5 * (E_plus - E_minus)
                grad[i] = dE

            # Gradient descent on L_G == -E => theta <- theta + lr_g * grad(E)
            theta = theta + float(lr_g) * grad

            lossG = float(-np.dot(G.probs(theta), v))

        history["loss_D"].append(float(lossD.detach().cpu().item()))
        history["loss_G"].append(float(lossG))

        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"[QGAN] epoch {ep:4d}/{epochs}  lossD={history['loss_D'][-1]:.4f}  lossG={history['loss_G'][-1]:.4f}")

    timings["train_total"] = time.perf_counter() - t_start
    timings["epochs"] = int(epochs)

    return QGANResult(generator_params=theta, history=history, discretization=meta, timings=timings)


def sample_qgan_1d(result: QGANResult, n_samples: int, *, gen_layers: int = 2) -> np.ndarray:
    """
    Sample real-valued outputs from a trained QGANResult (1D).
    """
    n_qubits = int(result.discretization["n_qubits"])
    G = QuantumGenerator(n_qubits=n_qubits, layers=gen_layers)
    p = G.probs(result.generator_params)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(2 ** n_qubits), size=int(n_samples), replace=True, p=p)
    return _undiscretize_1d(idx, result.discretization)
