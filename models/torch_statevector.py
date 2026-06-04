from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import Statevector


def encoded_state_matrix(encoding, X: np.ndarray) -> np.ndarray:
    """Return encoded statevectors for X as a complex matrix."""
    X = np.asarray(X, dtype=float)
    states = []
    for i in range(X.shape[0]):
        states.append(Statevector.from_instruction(encoding.build(X[i])).data)
    return np.asarray(states, dtype=np.complex128)


def _torch():
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific.
        raise RuntimeError("PyTorch is required for optimizer='backprop'. Install torch first.") from exc


def to_torch_states(states: np.ndarray):
    torch = _torch()
    return torch.tensor(np.asarray(states, dtype=np.complex128), dtype=torch.complex128)


def to_torch_labels(y: np.ndarray):
    torch = _torch()
    return torch.tensor(np.asarray(y, dtype=float), dtype=torch.float64)


def make_parameter(theta0: np.ndarray):
    torch = _torch()
    return torch.nn.Parameter(torch.tensor(np.asarray(theta0, dtype=float), dtype=torch.float64))


def bce_loss_from_prob(p, y, eps: float = 1e-8):
    p = p.clamp(float(eps), 1.0 - float(eps))
    return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()


def probability_one(states, readout: int):
    torch = _torch()
    n = int(round(np.log2(states.shape[1])))
    idx = torch.arange(states.shape[1], device=states.device)
    mask = ((idx >> int(readout)) & 1).to(torch.bool)
    probs = states.abs().square()
    return probs[:, mask].sum(dim=1)


def apply_ry(states, theta, qubit: int):
    torch = _torch()
    idx0, idx1 = _bit_pair_indices(int(states.shape[1]), int(qubit))
    idx0 = torch.tensor(idx0, device=states.device, dtype=torch.long)
    idx1 = torch.tensor(idx1, device=states.device, dtype=torch.long)

    out = states.clone()
    a0 = states.index_select(1, idx0)
    a1 = states.index_select(1, idx1)
    c = torch.cos(theta / 2.0).to(states.dtype)
    s = torch.sin(theta / 2.0).to(states.dtype)
    out[:, idx0] = c * a0 - s * a1
    out[:, idx1] = s * a0 + c * a1
    return out


def apply_rz(states, theta, qubit: int):
    torch = _torch()
    idx0, idx1 = _bit_pair_indices(int(states.shape[1]), int(qubit))
    idx0 = torch.tensor(idx0, device=states.device, dtype=torch.long)
    idx1 = torch.tensor(idx1, device=states.device, dtype=torch.long)

    out = states.clone()
    phase0 = torch.exp((-0.5j * theta).to(states.dtype))
    phase1 = torch.exp((0.5j * theta).to(states.dtype))
    out[:, idx0] = phase0 * states.index_select(1, idx0)
    out[:, idx1] = phase1 * states.index_select(1, idx1)
    return out


def apply_cnot(states, control: int, target: int):
    torch = _torch()
    inv = _cnot_inverse_permutation(int(states.shape[1]), int(control), int(target))
    inv_t = torch.tensor(inv, device=states.device, dtype=torch.long)
    return states.index_select(1, inv_t)


def apply_qnn_ansatz(states, theta, *, num_qubits: int, depth: int):
    idx = 0
    out = states
    for _ in range(int(depth)):
        for q in range(int(num_qubits)):
            out = apply_ry(out, theta[idx], q)
            idx += 1
            out = apply_rz(out, theta[idx], q)
            idx += 1
        if int(num_qubits) >= 2:
            for q in range(int(num_qubits) - 1):
                out = apply_cnot(out, q, q + 1)
            out = apply_cnot(out, int(num_qubits) - 1, 0)
    return out


def apply_qcnn_ansatz(states, theta, *, num_qubits: int, stages: int):
    out = states
    active = list(range(int(num_qubits)))
    idx = 0

    for _ in range(int(stages)):
        if len(active) < 2:
            break
        new_active: List[int] = []
        for i in range(0, len(active) - 1, 2):
            q0 = active[i]
            q1 = active[i + 1]

            out = apply_ry(out, theta[idx], q0)
            idx += 1
            out = apply_ry(out, theta[idx], q1)
            idx += 1
            out = apply_rz(out, theta[idx], q0)
            idx += 1
            out = apply_rz(out, theta[idx], q1)
            idx += 1
            out = apply_cnot(out, q0, q1)
            out = apply_rz(out, theta[idx], q1)
            idx += 1
            out = apply_cnot(out, q0, q1)

            out = apply_cnot(out, q1, q0)
            out = apply_ry(out, theta[idx], q0)
            idx += 1
            out = apply_rz(out, theta[idx], q0)
            idx += 1
            out = apply_cnot(out, q1, q0)

            new_active.append(q0)
        active = new_active

    if active and idx + 1 < theta.numel():
        q = active[0]
        out = apply_ry(out, theta[idx], q)
        idx += 1
        out = apply_rz(out, theta[idx], q)

    return out


@lru_cache(maxsize=128)
def _bit_pair_indices(dim: int, qubit: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    idx0 = []
    idx1 = []
    bit = 1 << int(qubit)
    for i in range(int(dim)):
        if (i & bit) == 0:
            idx0.append(i)
            idx1.append(i | bit)
    return tuple(idx0), tuple(idx1)


@lru_cache(maxsize=128)
def _cnot_inverse_permutation(dim: int, control: int, target: int) -> Tuple[int, ...]:
    inv = [0] * int(dim)
    cbit = 1 << int(control)
    tbit = 1 << int(target)
    for i in range(int(dim)):
        j = i ^ tbit if (i & cbit) else i
        inv[j] = i
    return tuple(inv)
