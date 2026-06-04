from __future__ import annotations

from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
    _HAVE_AER = True
except Exception:
    AerSimulator = None
    _HAVE_AER = False


def _statevectors(encoding, X: np.ndarray) -> np.ndarray:
    """Return a (n_samples, 2**n_qubits) complex array of statevectors."""
    X = np.asarray(X, dtype=float)
    states = []
    for i in range(X.shape[0]):
        qc: QuantumCircuit = encoding.build(X[i])
        states.append(Statevector.from_instruction(qc).data)
    return np.asarray(states, dtype=complex)


def _fidelity_from_statevectors(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    amp = np.vdot(psi_a, psi_b)
    return float(np.abs(amp) ** 2)


def _prob_allzero_counts(
    encoding,
    a: np.ndarray,
    b: np.ndarray,
    *,
    shots: int,
    backend: Optional[object] = None,
    optimization_level: int = 1,
) -> float:
    """
    Overlap test by applying U(b) then U(a)^† and measuring the probability of |0...0>.

    This is a common *kernel* estimator used in quantum kernel methods. It is *not*
    the SWAP test; it is the "all-zero after uncompute" method.
    """
    if not _HAVE_AER:
        raise RuntimeError("qiskit-aer not available; cannot run shots-based kernel.")

    U_a: QuantumCircuit = encoding.build(a)
    U_b: QuantumCircuit = encoding.build(b)

    n = U_a.num_qubits
    qc = QuantumCircuit(n)
    qc.append(U_b.to_instruction(), range(n))
    qc.append(U_a.to_instruction().inverse(), range(n))
    qc.measure_all()

    be = backend or AerSimulator()
    tqc = transpile(qc, be, optimization_level=optimization_level)
    res = be.run(tqc, shots=int(shots)).result()
    counts = res.get_counts()
    p0 = counts.get("0" * n, 0) / float(shots)
    return float(p0)


def kernel_entry_allzero(
    encoding,
    a: np.ndarray,
    b: np.ndarray,
    *,
    shots: Optional[int] = None,
    backend: Optional[object] = None,
) -> float:
    """
    Compute kernel entry K(a,b) as fidelity between encoded states.

    - If shots is None (default): exact fidelity via statevector simulation.
    - If shots is an int: estimate using the all-zero overlap circuit on a qasm simulator.
    """
    if shots is None:
        U_a: QuantumCircuit = encoding.build(a)
        U_b: QuantumCircuit = encoding.build(b)
        psi_a = Statevector.from_instruction(U_a).data
        psi_b = Statevector.from_instruction(U_b).data
        return _fidelity_from_statevectors(psi_a, psi_b)

    return _prob_allzero_counts(encoding, a, b, shots=int(shots), backend=backend)


def kernel_matrix(
    encoding,
    X: np.ndarray,
    *,
    Y: Optional[np.ndarray] = None,
    shots: Optional[int] = None,
    backend: Optional[object] = None,
) -> np.ndarray:
    """
    Compute Gram matrix K where K[i,j] = K(x_i, y_j).

    Parameters
    ----------
    encoding:
        An encoding object with .build(x) -> QuantumCircuit.
    X:
        Array of shape (n, d)
    Y:
        Optional array of shape (m, d). If None, compute symmetric Gram matrix for X.
    shots:
        None for exact statevector fidelity; int for sampling-based estimate.
    backend:
        Optional Qiskit backend for shots-based execution.

    Returns
    -------
    K: ndarray
        Shape (n, m) if Y provided, else (n, n).
    """
    X = np.asarray(X, dtype=float)
    if Y is not None:
        Y = np.asarray(Y, dtype=float)

    if shots is None:
        # Vectorized exact fidelity using statevectors.
        svX = _statevectors(encoding, X)
        if Y is None:
            inner = svX.conj() @ svX.T
            K = np.abs(inner) ** 2
            # Ensure exact symmetry (numerical)
            return 0.5 * (K + K.T)
        svY = _statevectors(encoding, Y)
        inner = svX.conj() @ svY.T
        return (np.abs(inner) ** 2).astype(float)

    # Shots-based: explicit pairwise evaluations (expensive but matches hardware flow).
    n = X.shape[0]
    if Y is None:
        K = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                kij = kernel_entry_allzero(encoding, X[i], X[j], shots=shots, backend=backend)
                K[i, j] = K[j, i] = kij
        return K

    m = Y.shape[0]
    K = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_entry_allzero(encoding, X[i], Y[j], shots=shots, backend=backend)
    return K
