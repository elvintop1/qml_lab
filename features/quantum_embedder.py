# qml_lab/features/quantum_embedder.py
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np

# We intentionally use your encodings registry and Qiskit statevector here
from qml_lab.encodings.registry import get_encoding_cls
from qiskit.quantum_info import Statevector

def _upper_tri_indices(m: int) -> Tuple[np.ndarray, np.ndarray]:
    triu = np.triu_indices(m, k=1)
    return triu[0], triu[1]

def _expvals_from_state(psi: np.ndarray, n_qubits: int, *, want_x: bool, want_y: bool, want_z: bool, want_zz: bool) -> np.ndarray:
    """Compute <X_i>, <Y_i>, <Z_i> and <Z_i Z_j> from a statevector (exact, no sampling)."""
    idx = np.arange(psi.size, dtype=np.int64)
    out: List[np.ndarray] = []
    probs = np.abs(psi) ** 2

    # <Z_i>
    Z = None
    if want_z or want_zz:
        z_list = []
        for q in range(n_qubits):
            bit = (idx >> q) & 1
            plus = probs[bit == 0].sum()
            minus = probs[bit == 1].sum()
            z_list.append(plus - minus)
        Z = np.array(z_list, dtype=np.float64)
        if want_z:
            out.append(Z[None, :])

    # <X_i>, <Y_i>
    if want_x or want_y:
        for q in range(n_qubits):
            flip = idx ^ (1 << q)
            if want_x:
                xexp = np.vdot(psi, psi[flip])
                out.append(np.array([[np.real(xexp)]]))
            if want_y:
                bit = (idx >> q) & 1
                factor = 1j * (1 - 2 * bit)  
                yexp = np.vdot(psi, factor * psi[flip])
                out.append(np.array([[np.real(yexp)]]))

    if want_zz and n_qubits >= 2 and Z is not None:
        i, j = _upper_tri_indices(n_qubits)
        ZZ = (Z[:, None] * Z[None, :])[i, j]
        out.append(ZZ[None, :])

    return np.concatenate(out, axis=1).astype(np.float32) if out else np.zeros((1, 0), np.float32)


def quantum_embed_features(
    X: np.ndarray,
    *,
    encoding_name: str,
    encoding_params: Optional[dict] = None,
    features_per_qubit: Optional[int] = None,
    observables: str = "z",    # "z" | "xyz" | "z+zz" | "xyz+zz"
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got {X.shape}")
    n, d = X.shape

    enc_key = encoding_name.lower().replace("-", "_")
    params = dict(encoding_params or {})
    fpp = 2 if (features_per_qubit is None and enc_key in ("denseangle", "dense_angle")) else (features_per_qubit or 1)

    if enc_key != "amplitude" and d % fpp != 0:
        raise ValueError(f"{encoding_name}: features_per_qubit={fpp} but d={d}; need d % fpp == 0 for non-amplitude encodings.")

    # number of qubits
    if enc_key == "amplitude":
        num_qubits = int(np.ceil(np.log2(max(1, d))))
    else:
        num_qubits = d // fpp

    Enc = get_encoding_cls(enc_key)
    encoder = Enc(num_qubits=num_qubits, **params)

    obs = observables.lower()
    want_x = "x" in obs
    want_y = "y" in obs
    want_z = "z" in obs
    want_zz = "zz" in obs

    rows: List[np.ndarray] = []
    for i in range(n):
        qc = encoder.build(X[i])
        psi = Statevector.from_instruction(qc).data
        feats = _expvals_from_state(psi, num_qubits, want_x=want_x, want_y=want_y, want_z=want_z, want_zz=want_zz)
        rows.append(feats)

    return np.vstack(rows).astype(np.float32)
