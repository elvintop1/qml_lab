from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from qiskit.quantum_info import Statevector

from ..encodings.helpers import build_encoding
from ..kernels import kernel_matrix
from .protocols import RunResult


def _build_swap_test_circuit(enc, a: np.ndarray, b: np.ndarray):
    """Representative swap-test circuit.

    Circuit width = 1 + 2*n where n is the number of qubits used by the encoding.
    Measuring ancilla yields P(0) = (1 + |<ψ(a)|ψ(b)>|^2)/2.
    """
    from qiskit import QuantumCircuit

    U_a = enc.build(a)
    U_b = enc.build(b)
    n = U_a.num_qubits
    qc = QuantumCircuit(1 + 2 * n, 1, name="SwapTest")
    anc = 0
    reg_a = list(range(1, 1 + n))
    reg_b = list(range(1 + n, 1 + 2 * n))

    qc.h(anc)
    qc.compose(U_a, reg_a, inplace=True)
    qc.compose(U_b, reg_b, inplace=True)
    for i in range(n):
        qc.cswap(anc, reg_a[i], reg_b[i])
    qc.h(anc)
    qc.measure(anc, 0)
    return qc


def _swap_test_fidelity_pair(enc, a: np.ndarray, b: np.ndarray, *, shots: Optional[int] = None) -> float:
    """Return fidelity |<ψ(a)|ψ(b)>|^2.

    - If shots is None: exact via statevector inner product (fast on simulator)
    - Else: estimate via swap-test circuit with `shots`
    """
    if shots is None:
        va = Statevector.from_instruction(enc.build(a)).data
        vb = Statevector.from_instruction(enc.build(b)).data
        return float(np.abs(np.vdot(va, vb)) ** 2)

    # Shots-based swap test
    try:
        from qiskit_aer import AerSimulator
    except Exception as e:
        raise RuntimeError("qiskit-aer is required for shots-based swap-test execution.") from e

    be = AerSimulator()
    qc = _build_swap_test_circuit(enc, a, b)
    res = be.run(qc, shots=int(shots)).result()
    counts = res.get_counts()
    p0 = counts.get("0", 0) / float(shots)
    F = 2.0 * p0 - 1.0
    return float(np.clip(F, 0.0, 1.0))


def _swap_test_fidelity_matrix(enc, X: np.ndarray, Y: np.ndarray, *, shots: Optional[int] = None) -> np.ndarray:
    """Compute fidelity matrix F[i,j] = |<ψ(X[i])|ψ(Y[j])>|^2."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if shots is None:
        # Fast path: prepare each state once, then inner products classically.
        sv_X = np.asarray([Statevector.from_instruction(enc.build(x)).data for x in X], dtype=complex)
        sv_Y = np.asarray([Statevector.from_instruction(enc.build(y)).data for y in Y], dtype=complex)
        inner = sv_X.conj() @ sv_Y.T
        return (np.abs(inner) ** 2).astype(float)

    # Shots-based: do not attempt to batch huge grids; iterate.
    F = np.zeros((len(X), len(Y)), dtype=float)
    for i in range(len(X)):
        for j in range(len(Y)):
            F[i, j] = _swap_test_fidelity_pair(enc, X[i], Y[j], shots=shots)
    return F


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0 or n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    target = np.maximum(1, np.floor(props * n).astype(int))
    while target.sum() < n:
        target[np.argmax(props)] += 1
    while target.sum() > n:
        i = np.argmax(target)
        if target[i] > 1:
            target[i] -= 1
        else:
            break

    idx = []
    for c, k in zip(classes, target):
        pool = np.where(y == c)[0]
        k = min(int(k), pool.size)
        idx.append(rng.choice(pool, size=k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _scale_features(Xtr_raw: np.ndarray, Xte_raw: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0.0, 2 * np.pi))
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)
    return Xtr, Xte, scaler


def _quantum_distance_from_fidelity(F: np.ndarray) -> np.ndarray:
    """
    Distance derived from fidelity:
      d_Q(x,z) = sqrt(2 * (1 - sqrt(F))).
    """
    F = np.clip(F, 0.0, 1.0)
    return np.sqrt(2.0 * (1.0 - np.sqrt(F)))




# ---------------------------------------------------------------------------
# Adaptive prototype memory for QKNN
# ---------------------------------------------------------------------------


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix with numerical clipping."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays.")
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    D = aa + bb - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)


def _latent_from_encoding(enc, X: np.ndarray) -> Optional[np.ndarray]:
    """Return HA-SAGE-CMTSD latent margins when available.

    QKNN prototype selection is a classical memory-compression step, so using
    the fitted latent margin geometry is cheap and does not change the quantum
    overlap computation used at prediction time.
    """
    if not hasattr(enc, "_project"):
        return None
    rows = []
    try:
        for row in np.asarray(X, dtype=float):
            out = enc._project(row)
            if isinstance(out, tuple) and len(out) >= 3:
                # margin is the tree/metric-aware coordinate; it is better for
                # boundary prototypes than raw u or saturated z.
                rows.append(np.asarray(out[1], dtype=float))
            elif isinstance(out, tuple) and len(out) >= 1:
                rows.append(np.asarray(out[0], dtype=float))
            else:
                return None
    except Exception:
        return None
    if not rows:
        return None
    U = np.asarray(rows, dtype=float)
    if U.ndim != 2 or U.shape[0] != len(X):
        return None
    return U


def _knn_predict_from_distance(D: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Deterministic kNN vote over a precomputed distance matrix."""
    D = np.asarray(D, dtype=float)
    y_train = np.asarray(y_train)
    k = max(1, min(int(k), D.shape[1]))
    nn = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    pred = []
    for local in nn:
        labels = y_train[local]
        vals, counts = np.unique(labels, return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return np.asarray(pred)


def _loo_knn_latent_score(U: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-out kNN score in latent space."""
    if U is None or len(U) < 3 or np.unique(y).size < 2:
        return 0.0
    D = _pairwise_sqdist(U, U)
    np.fill_diagonal(D, np.inf)
    pred = _knn_predict_from_distance(D, y, k=min(int(k), len(U) - 1))
    return float(np.mean(pred == y))


def _prototype_latent_score(U: np.ndarray, y: np.ndarray, proto: List[int], k: int) -> float:
    """Training-set score when every sample queries only the prototype memory."""
    if U is None or not proto:
        return 0.0
    proto = sorted({int(i) for i in proto if 0 <= int(i) < len(U)})
    if not proto:
        return 0.0
    P = U[np.asarray(proto, dtype=int)]
    yp = y[np.asarray(proto, dtype=int)]
    D = _pairwise_sqdist(U, P)
    # Avoid giving zero-distance self-matches to samples that are themselves
    # prototypes; otherwise the train score is over-optimistic and the prototype
    # budget stays too small on hard synthetic datasets.
    proto_pos = {idx: j for j, idx in enumerate(proto)}
    for i, j in proto_pos.items():
        D[i, j] = np.inf
    # If a class has only one prototype, the above exclusion can leave a row
    # entirely finite via other-class prototypes; this is intended and exposes
    # the weakness of an overly small memory.
    pred = _knn_predict_from_distance(D, yp, k=min(int(k), len(proto)))
    return float(np.mean(pred == y))


def _class_boundary_scores(U: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Small score means closer to a class boundary in latent space."""
    if idx.size == 0:
        return np.asarray([], dtype=float)
    Xc = U[idx]
    yc = y[idx]
    out = np.zeros(idx.size, dtype=float)
    all_classes = np.unique(y)
    for loc, global_i in enumerate(idx):
        same = np.where(y == y[global_i])[0]
        other = np.where(y != y[global_i])[0]
        if same.size <= 1 or other.size == 0:
            out[loc] = float(np.min(np.abs(U[global_i]))) if U.shape[1] else 0.0
            continue
        d_same = _pairwise_sqdist(U[global_i:global_i + 1], U[same])[0]
        d_same[same == global_i] = np.inf
        d_other = _pairwise_sqdist(U[global_i:global_i + 1], U[other])[0]
        out[loc] = float(np.min(d_other) - np.min(d_same))
    return out


def _select_adaptive_latent_prototypes(
    U: np.ndarray,
    y: np.ndarray,
    *,
    base_indices: List[int],
    k: int,
    seed: int,
    keep_ratio: float,
    min_per_class: int,
    max_per_class: Optional[int],
) -> List[int]:
    """Class-balanced prototype memory with centroid, boundary and coverage points.

    The previous implementation used only 2 prototypes per class in many runs.
    That saved calls but caused large QKNN failures on synthetic_16/synthetic_32.
    This selector scales the budget with the class size and k, while preserving
    the important prototype types already discovered by C-MTSD.
    """
    U = np.asarray(U, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed) + 4103)
    n = len(y)
    if n == 0 or np.unique(y).size < 2:
        return []

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    out = {int(i) for i in base_indices if 0 <= int(i) < n}
    min_pc_global = max(int(min_per_class), int(k) + 1, 2)

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_c = int(idx.size)
        if n_c == 0:
            continue
        target = max(min_pc_global, int(np.ceil(keep_ratio * n_c)), int(np.ceil(np.sqrt(n_c))))
        if max_per_class is not None and int(max_per_class) > 0:
            target = min(target, int(max_per_class))
        target = min(target, n_c)

        chosen = [int(i) for i in sorted(out) if 0 <= int(i) < n and y[int(i)] == c]

        # 1) Class centroid medoid.
        if len(chosen) < target:
            Xc = U[idx]
            centroid = np.mean(Xc, axis=0)
            d_cent = np.sum((Xc - centroid[None, :]) ** 2, axis=1)
            cand = int(idx[int(np.argmin(d_cent))])
            if cand not in chosen:
                chosen.append(cand)

        # 2) Boundary candidates: nearest to the opposite class, not just nearest
        # to zero margin.  This protects checkerboard/XOR-like synthetic tasks.
        if len(chosen) < target:
            bscore = _class_boundary_scores(U, y, idx)
            for local in np.argsort(bscore):
                cand = int(idx[int(local)])
                if cand not in chosen:
                    chosen.append(cand)
                if len(chosen) >= target:
                    break

        # 3) Farthest-first coverage within each class.
        if len(chosen) < target:
            if chosen:
                dist_best = np.min(_pairwise_sqdist(U[idx], U[np.asarray(chosen, dtype=int)]), axis=1)
            else:
                # Reproducible seed point when the class has no candidate yet.
                first = int(rng.integers(0, n_c))
                chosen.append(int(idx[first]))
                dist_best = _pairwise_sqdist(U[idx], U[[chosen[-1]]]).ravel()
            while len(chosen) < target:
                jitter = rng.uniform(0.0, 1e-12, size=dist_best.shape)
                nxt = int(idx[int(np.argmax(dist_best + jitter))])
                if nxt not in chosen:
                    chosen.append(nxt)
                dist_best = np.minimum(dist_best, _pairwise_sqdist(U[idx], U[[nxt]]).ravel())
                if len(set(chosen)) >= n_c:
                    break

        out.update(chosen[:target])

    return sorted(out)


def _maybe_apply_adaptive_prototypes(enc, Xtr: np.ndarray, ytr: np.ndarray, cfg: "QKNNConfig", seed: int):
    """Return possibly-compressed training memory and diagnostics."""
    requested = bool(getattr(cfg, "use_encoding_prototypes", False)) or bool(
        cfg.encoding_params.get("use_prototypes", False)
    )
    diagnostics = {
        "prototype_policy": str(getattr(cfg, "prototype_policy", "adaptive")),
        "used_encoding_prototypes": False,
        "prototype_indices": [],
        "n_prototypes": 0,
        "prototype_train_score": None,
        "prototype_full_train_score": None,
        "prototype_fallback_reason": "not_requested" if not requested else None,
    }
    if not requested:
        return Xtr, ytr, False, [], diagnostics

    policy = str(getattr(cfg, "prototype_policy", "adaptive") or "adaptive").lower().strip()
    if policy in {"off", "none", "false", "0"}:
        diagnostics["prototype_fallback_reason"] = "policy_off"
        return Xtr, ytr, False, [], diagnostics

    base = []
    if hasattr(enc, "prototype_indices"):
        base = [int(i) for i in getattr(enc, "prototype_indices", []) if 0 <= int(i) < len(Xtr)]
    base = sorted(set(base))

    if policy in {"fixed", "encoding", "cmtsd"}:
        proto = base
    else:
        U = _latent_from_encoding(enc, Xtr)
        if U is None:
            proto = base
            diagnostics["prototype_fallback_reason"] = "no_latent_projection"
        else:
            min_pc = int(getattr(cfg, "min_prototypes_per_class", 0) or 0)
            if min_pc <= 0:
                min_pc = max(int(cfg.k) + 1, 4)
            proto = _select_adaptive_latent_prototypes(
                U,
                ytr,
                base_indices=base,
                k=int(cfg.k),
                seed=int(seed),
                keep_ratio=float(getattr(cfg, "prototype_keep_ratio", 0.35)),
                min_per_class=min_pc,
                max_per_class=getattr(cfg, "max_prototypes_per_class", None),
            )
            full_score = _loo_knn_latent_score(U, ytr, k=int(cfg.k))
            proto_score = _prototype_latent_score(U, ytr, proto, k=int(cfg.k))
            diagnostics["prototype_train_score"] = float(proto_score)
            diagnostics["prototype_full_train_score"] = float(full_score)

            min_score = float(getattr(cfg, "prototype_min_train_score", 0.88))
            max_gap = float(getattr(cfg, "prototype_max_train_gap", 0.08))
            if policy in {"adaptive", "auto", "safe"} and proto:
                hard_failure = proto_score < min_score and (full_score - proto_score) > max_gap
                if hard_failure:
                    # Try one larger memory before falling back.  This keeps the
                    # resource-saving story while avoiding the 4-prototype collapse.
                    proto2 = _select_adaptive_latent_prototypes(
                        U,
                        ytr,
                        base_indices=proto,
                        k=int(cfg.k),
                        seed=int(seed) + 17,
                        keep_ratio=max(float(getattr(cfg, "prototype_keep_ratio", 0.35)), 0.65),
                        min_per_class=max(min_pc, 2 * int(cfg.k) + 1),
                        max_per_class=getattr(cfg, "max_prototypes_per_class", None),
                    )
                    proto2_score = _prototype_latent_score(U, ytr, proto2, k=int(cfg.k))
                    diagnostics["prototype_train_score_after_expand"] = float(proto2_score)
                    if proto2_score >= min_score or (full_score - proto2_score) <= max_gap:
                        proto = proto2
                        diagnostics["prototype_fallback_reason"] = "expanded_memory"
                    else:
                        diagnostics["prototype_fallback_reason"] = "train_score_guard_fallback_full_memory"
                        return Xtr, ytr, False, [], diagnostics

    proto = sorted({int(i) for i in proto if 0 <= int(i) < len(Xtr)})
    if proto and set(np.unique(ytr[proto]).tolist()) == set(np.unique(ytr).tolist()):
        idx = np.asarray(proto, dtype=int)
        diagnostics["used_encoding_prototypes"] = True
        diagnostics["prototype_indices"] = proto
        diagnostics["n_prototypes"] = int(len(proto))
        diagnostics["prototype_fallback_reason"] = diagnostics.get("prototype_fallback_reason") or "used"
        return Xtr[idx], ytr[idx], True, proto, diagnostics

    diagnostics["prototype_fallback_reason"] = "missing_class_in_prototypes"
    return Xtr, ytr, False, [], diagnostics


@dataclass
class QKNNConfig:
    encoding_name: str
    encoding_params: Dict
    feat_idx: List[int]
    k: int = 3
    shots: Optional[int] = None
    vote: str = "uniform"  # uniform | distance_weighted | fidelity_weighted

    # --- Variants (Phase-1 thesis compatibility) ---
    # "swap_test": estimate |<ψ(x)|ψ(z)>|^2 via swap-test (or exact statevector if shots=None)
    # "grover_style": same distance oracle, neighbour search conceptually accelerated (still classical here)
    variant: str = "swap_test"

    max_train: Optional[int] = None
    max_test: Optional[int] = None

    # HA-SAGE-CMTSD support: after the encoding is fitted on the training split,
    # use its class-balanced prototype_indices as the neighbour memory.  This
    # reduces QKNN kernel calls from n_test * n_train to n_test * n_prototypes.
    use_encoding_prototypes: bool = False

    # Prototype memory control.  "fixed" uses only enc.prototype_indices.
    # "adaptive" expands that memory in latent space and falls back to full
    # memory when the train-split latent kNN score drops too much.
    prototype_policy: str = "adaptive"  # off | fixed | adaptive | safe
    prototype_keep_ratio: float = 0.35
    min_prototypes_per_class: int = 0
    max_prototypes_per_class: Optional[int] = None
    prototype_min_train_score: float = 0.88
    prototype_max_train_gap: float = 0.08


def run_qknn_train_test(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: QKNNConfig,
    test_size: float = 0.30,
    seed: int = 42,
    *,
    return_result: bool = False,
    capture_circuits: bool = False,
):
    """
    QKNN variants (thesis Phase-1 compatibility):

    - Swap-test QKNN: estimate |<ψ(x)|ψ(z)>|^2 via swap test.
      * shots=None: exact fidelities via statevector inner products (fastest on simulator)
      * shots=int : Monte-Carlo estimate using the swap-test circuit

    - Grover-style QKNN: uses the same distance oracle as swap-test QKNN.
      In this codebase we still perform neighbour search classically; the Grover label is
      included for reporting consistency with the thesis architecture table.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    # Variant selection
    vraw = str(getattr(cfg, "variant", "swap_test") or "swap_test").lower().strip()
    vkey = vraw.replace("-", "").replace("_", "").replace(" ", "")
    if vkey in {"swaptest", "swap"}:
        variant = "swap_test"
    elif vkey in {"grover", "groverstyle", "groverknn"}:
        variant = "grover_style"
    else:
        raise ValueError(f"Unknown QKNN variant: {cfg.variant!r}. Use 'swap_test' or 'grover_style'.")

    Xsel = X_raw[:, cfg.feat_idx]
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xsel, y, test_size=test_size, random_state=seed, stratify=y
    )

    if cfg.max_train is not None:
        Xtr_raw, ytr = _stratified_subsample(Xtr_raw, ytr, int(cfg.max_train), seed=seed)
    if cfg.max_test is not None:
        Xte_raw, yte = _stratified_subsample(Xte_raw, yte, int(cfg.max_test), seed=seed + 1)

    Xtr, Xte, scaler = _scale_features(Xtr_raw, Xte_raw)

    enc, num_qubits, _ = build_encoding(
        cfg.encoding_name,
        n_features=Xtr.shape[1],
        params=cfg.encoding_params,
        X_fit=Xtr,
        y_fit=ytr,
    )

    # Optional HA-SAGE-CMTSD / C-MTSD prototype memory for QKNN.
    # The new adaptive path fixes the observed 4-prototype collapse: it expands
    # the memory when latent train-set kNN would degrade and falls back to full
    # memory if compression is unsafe.
    Xtr, ytr, used_prototypes, prototype_indices, proto_diag = _maybe_apply_adaptive_prototypes(
        enc, Xtr, ytr, cfg, seed
    )

    timings = {}

    # Compute fidelity matrix between test and train
    t0 = time.perf_counter()
    F = _swap_test_fidelity_matrix(enc, Xte, Xtr, shots=cfg.shots)
    if cfg.shots is None:
        timings["fidelity_statevector"] = time.perf_counter() - t0
        # Simulator path: we only build |psi(x)> once per sample, then do inner products classically.
        evals_sim = int(len(Xtr) + len(Xte))
    else:
        timings["fidelity_shots"] = time.perf_counter() - t0
        # Hardware-like path: one swap-test circuit per (test, train) pair.
        evals_sim = int(len(Xte) * len(Xtr))

    # For hardware timing estimates, the algorithm needs (n_test * n_train) overlaps.
    evals_hw = int(len(Xte) * len(Xtr))

    # Distances and KNN vote
    t0 = time.perf_counter()
    D = _quantum_distance_from_fidelity(F)  # shape (n_te, n_tr)

    k = int(cfg.k)
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, D.shape[1])

    # argpartition for top-k smallest distances
    nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    nn_labels = ytr[nn_idx]  # shape (n_te, k)

    # Majority or weighted vote; tie -> choose smaller class label.
    # Distance/fidelity weighting is useful for QKNN because the nearest quantum
    # neighbour should count more than the kth neighbour when distances are not
    # uniform.  The default remains uniform for backwards-compatible baselines.
    vote = str(getattr(cfg, "vote", "uniform") or "uniform").lower().strip()
    yhat = np.zeros(len(yte), dtype=int)
    yprob = np.zeros(len(yte), dtype=float)

    for i in range(len(yte)):
        labels = nn_labels[i]
        local_idx = nn_idx[i]
        if vote in {"distance_weighted", "distance", "weighted"}:
            weights = 1.0 / (D[i, local_idx] + 1e-9)
        elif vote in {"fidelity_weighted", "fidelity"}:
            weights = F[i, local_idx]
        else:
            weights = np.ones_like(labels, dtype=float)

        # Probability proxy for class "1" (only meaningful for binary {0,1} tasks).
        denom = float(np.sum(weights)) if float(np.sum(weights)) > 0.0 else 1.0
        yprob[i] = float(np.sum(weights[labels == 1]) / denom)

        vals = np.unique(labels)
        scores = np.asarray([float(np.sum(weights[labels == v])) for v in vals], dtype=float)
        yhat[i] = int(vals[np.argmax(scores)])

    timings["knn_vote"] = time.perf_counter() - t0

    acc = float(accuracy_score(yte, yhat))
    cm = confusion_matrix(yte, yhat)

    model_name = "QKNN (SwapTest)" if variant == "swap_test" else "QKNN (GroverStyle)"
    # Swap-test circuit uses an ancilla + two copies of the encoding register
    total_qubits = int(1 + 2 * num_qubits)

    proto_tag = f" prototypes={len(prototype_indices)}" if used_prototypes else ""
    tag = f"{model_name} enc={cfg.encoding_name} k={cfg.k} shots={cfg.shots} qubits={total_qubits}{proto_tag}"

    if not return_result:
        return acc, cm, tag

    circuits = None
    if capture_circuits:
        try:
            a = Xtr[0]
            b = Xtr[min(1, len(Xtr) - 1)]
            circuits = [_build_swap_test_circuit(enc, a, b)]
        except Exception:
            circuits = None

    result = RunResult(
        y_true=yte,
        y_pred=yhat,
        y_prob=np.vstack([1.0 - yprob, yprob]).T,  # binary probs
        K_train=None,
        K_test=None,
        y_train=None,
        circuits=circuits,
        model_name=model_name,
        encoding_name=cfg.encoding_name,
        encoding_params=cfg.encoding_params,
        feat_idx=cfg.feat_idx,
        num_params=None,
        shots=cfg.shots,
        evals=evals_hw,
        timings=timings,
        extras={
            "tag": tag,
            "confusion_matrix": cm,
            "num_qubits": total_qubits,
            "encoding_qubits": int(num_qubits),
            "k": cfg.k,
            "variant": variant,
            "vote": vote,
            "used_encoding_prototypes": bool(used_prototypes),
            "prototype_indices": prototype_indices,
            "n_prototypes": int(len(prototype_indices)),
            **proto_diag,
            "evals_sim": int(evals_sim),
            "evals_hw": int(evals_hw),
            # For thesis discussion: Grover-style nearest-neighbour search would ideally reduce the
            # number of oracle queries. We do not implement Grover's minimum finding here.
            "conceptual_evals_hw_grover": int(len(Xte) * int(np.ceil(np.sqrt(len(Xtr))))) if len(Xtr) > 0 else 0,
        },
    )
    return acc, cm, tag, result
