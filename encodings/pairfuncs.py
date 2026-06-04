import numpy as np


def pair_phi(kind: str):
    """Pairwise phase functions used by IQP/Hamiltonian-style encodings."""
    k = str(kind or "prod").lower().strip()
    if k == "prod":
        return lambda x, i, j: 0.5 * (np.pi - x[i]) * (np.pi - x[j])
    if k == "cos":
        return lambda x, i, j: (np.pi / 3.0) * np.cos(x[i]) * np.cos(x[j])
    if k in {"gauss", "gaussian", "rbf"}:
        # Bug fix: the exponent must be negative.  The previous positive
        # exponent created huge phases for distant points and made the kernel
        # nearly random after modulo 2π wrapping.
        sigma2 = 8.0 / np.log(np.pi)
        return lambda x, i, j: np.exp(-(abs(x[i] - x[j]) ** 2) / sigma2)
    if k in {"lin", "linear", "zz", "product"}:
        return lambda x, i, j: 0.5 * x[i] * x[j]
    raise ValueError(f"Unknown pair_phi kind: {kind}")
