from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from ..encodings.helpers import build_encoding
from ..kernels import kernel_matrix, nearest_psd, center_train_test
from .protocols import RunResult


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample to n points while preserving class ratios as much as possible."""
    if n <= 0 or n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # target counts proportional to original
    props = counts / counts.sum()
    target = np.maximum(1, np.floor(props * n).astype(int))
    # adjust to exact n
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


@dataclass
class QSVMConfig:
    encoding_name: str
    encoding_params: Dict
    feat_idx: List[int]
    C: float = 1.0
    shots: Optional[int] = None

    # --- Variants (Phase-1 thesis compatibility) ---
    # "kernel_overlap": standard kernel QSVM (classical SVM on quantum kernel)
    # "ls_svm": LS-SVM / HHL-style linear system (solved classically here)
    variant: str = "kernel_overlap"
    lssvm_reg: float = 1e-3

    center: bool = True
    psd_fix: bool = False

    # Scaling controls
    max_train: Optional[int] = None
    max_test: Optional[int] = None

    # Optional Nyström approximation (useful when train set is larger than what full K can handle)
    nystrom_m: Optional[int] = None
    nystrom_reg: float = 1e-6


def _scale_features(Xtr_raw: np.ndarray, Xte_raw: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0.0, 2 * np.pi))
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)
    return Xtr, Xte, scaler


def _build_overlap_circuit(enc, a: np.ndarray, b: np.ndarray):
    """Representative overlap circuit used by the all-zero kernel estimator."""
    from qiskit import QuantumCircuit

    U_a = enc.build(a)
    U_b = enc.build(b)
    n = U_a.num_qubits
    qc = QuantumCircuit(n, name="KernelOverlap")
    qc.append(U_b.to_instruction(), range(n))
    qc.append(U_a.to_instruction().inverse(), range(n))
    qc.measure_all()
    return qc


def run_qsvm_train_test(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: QSVMConfig,
    test_size: float = 0.30,
    seed: int = 42,
    *,
    return_result: bool = False,
    capture_artifacts: bool = False,
    capture_circuits: bool = False,
):
    """
    Train/test QSVM using a quantum kernel (fidelity between encoded states).

    Notes on scaling:
      - Full kernel is O(n_train^2). Use cfg.max_train and/or cfg.nystrom_m to scale.
      - The quantum part is only the kernel evaluation; SVM fit/predict is classical.

    Returns
    -------
    (acc, cm, tag) or (acc, cm, tag, RunResult) if return_result=True
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    # -----------------------------
    # Variant selection (thesis Phase-1 compatibility)
    # -----------------------------
    vraw = str(getattr(cfg, "variant", "kernel_overlap") or "kernel_overlap").lower().strip()
    vkey = vraw.replace("-", "").replace("_", "").replace(" ", "")
    if vkey in {"kernel", "kerneloverlap", "qsvm", "svc"}:
        variant = "kernel_overlap"
    elif vkey in {"lssvm", "lssvmhhl", "hhl", "hhlstyle", "lssvmstyle", "lssvmhhlstyle"}:
        variant = "ls_svm"
    else:
        raise ValueError(f"Unknown QSVM variant: {cfg.variant!r}. Use 'kernel_overlap' or 'ls_svm'.")

    # Split
    Xsel = X_raw[:, cfg.feat_idx]
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xsel, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Optional subsampling for scalability
    if cfg.max_train is not None:
        Xtr_raw, ytr = _stratified_subsample(Xtr_raw, ytr, int(cfg.max_train), seed=seed)
    if cfg.max_test is not None:
        Xte_raw, yte = _stratified_subsample(Xte_raw, yte, int(cfg.max_test), seed=seed + 1)

    # Scale to [0, 2pi] for rotation-based encodings
    Xtr, Xte, scaler = _scale_features(Xtr_raw, Xte_raw)

    # Build encoding instance (handles denseangle fpp and amplitude qubit inference)
    enc, num_qubits, _ = build_encoding(
        cfg.encoding_name,
        n_features=Xtr.shape[1],
        params=cfg.encoding_params,
        X_fit=Xtr,
        y_fit=ytr,
    )

    # Kernel matrices
    timings = {}
    t0 = time.perf_counter()

    use_nystrom = (cfg.nystrom_m is not None) and (int(cfg.nystrom_m) > 0) and (int(cfg.nystrom_m) < len(Xtr))
    if use_nystrom:
        m = int(cfg.nystrom_m)
        rng = np.random.default_rng(seed)
        lm_idx = rng.choice(len(Xtr), size=m, replace=False)
        Xlm = Xtr[lm_idx]

        t1 = time.perf_counter()
        Kmm = kernel_matrix(enc, Xlm, shots=cfg.shots)
        timings["kernel_Kmm"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        Knm = kernel_matrix(enc, Xtr, Y=Xlm, shots=cfg.shots)
        timings["kernel_Knm"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        Kte_m = kernel_matrix(enc, Xte, Y=Xlm, shots=cfg.shots)
        timings["kernel_Kte_m"] = time.perf_counter() - t1

        # Nyström approximation
        t1 = time.perf_counter()
        reg = float(cfg.nystrom_reg)
        Kmm_reg = Kmm + reg * np.eye(m)
        Kmm_inv = np.linalg.pinv(Kmm_reg)
        Ktr = Knm @ Kmm_inv @ Knm.T
        Kte = Kte_m @ Kmm_inv @ Knm.T
        timings["kernel_nystrom_post"] = time.perf_counter() - t1

        # Eval accounting
        n_tr, n_te = len(Xtr), len(Xte)
        evals = (m * (m + 1) // 2) + (n_tr * m) + (n_te * m)
    else:
        t1 = time.perf_counter()
        Ktr = kernel_matrix(enc, Xtr, shots=cfg.shots)
        timings["kernel_train"] = time.perf_counter() - t1

        if cfg.psd_fix and cfg.shots is not None:
            t_fix = time.perf_counter()
            Ktr = nearest_psd(Ktr)
            timings["kernel_psd_fix"] = time.perf_counter() - t_fix

        t1 = time.perf_counter()
        Kte = kernel_matrix(enc, Xte, Y=Xtr, shots=cfg.shots)
        timings["kernel_test"] = time.perf_counter() - t1

        n_tr, n_te = len(Xtr), len(Xte)
        evals = (n_tr * (n_tr + 1) // 2) + (n_te * n_tr)

    timings["kernel_total"] = time.perf_counter() - t0

    # Centering (optional)
    if cfg.center:
        t0 = time.perf_counter()
        Ktr, Kte = center_train_test(Ktr, Kte)
        timings["kernel_center"] = time.perf_counter() - t0

    # -----------------------------
    # Classical decision layer (variant-dependent)
    # -----------------------------
    y_prob = None
    if variant == "kernel_overlap":
        # Standard SVM dual solved classically.
        clf = SVC(kernel="precomputed", C=float(cfg.C))
        t0 = time.perf_counter()
        clf.fit(Ktr, ytr)
        timings["svm_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        yhat = clf.predict(Kte)
        timings["svm_predict"] = time.perf_counter() - t0
    else:
        # LS-SVM / HHL-style linear system (we solve classically here):
        #   (K + λ I) α = y  (binary) or one-vs-rest (multi-class)
        # Decision: f(z) = Σ_i α_i K(z, x_i)
        lam = float(getattr(cfg, "lssvm_reg", 1e-3))
        A = Ktr + lam * np.eye(Ktr.shape[0])

        t0 = time.perf_counter()
        classes = np.unique(ytr)
        n_classes = int(len(classes))
        if n_classes == 2:
            # Map to {-1, +1} with +1 for the larger class label (np.unique is sorted).
            y_bin = np.where(ytr == classes[0], -1.0, 1.0).astype(float)
            alpha = np.linalg.solve(A, y_bin)
            timings["lssvm_solve"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            scores = (Kte @ alpha).astype(float)
            yhat = np.where(scores < 0.0, classes[0], classes[1]).astype(int)
            # Probability proxy via sigmoid (for ROC-AUC reporting)
            p_pos = 1.0 / (1.0 + np.exp(-scores))
            y_prob = np.vstack([1.0 - p_pos, p_pos]).T
            timings["lssvm_predict"] = time.perf_counter() - t0
        else:
            # One-vs-rest LS-SVM
            Y = -np.ones((len(ytr), n_classes), dtype=float)
            for j, c in enumerate(classes):
                Y[:, j] = np.where(ytr == c, 1.0, -1.0)
            Alpha = np.linalg.solve(A, Y)
            timings["lssvm_solve"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            scores = (Kte @ Alpha).astype(float)
            jhat = np.argmax(scores, axis=1)
            yhat = classes[jhat].astype(int)

            # Softmax probability proxy
            scores0 = scores - scores.max(axis=1, keepdims=True)
            exps = np.exp(scores0)
            y_prob = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)
            timings["lssvm_predict"] = time.perf_counter() - t0

    # Metrics
    acc = float(accuracy_score(yte, yhat))
    cm = confusion_matrix(yte, yhat)

    model_name = "QSVM (KernelOverlap)" if variant == "kernel_overlap" else "QSVM (LS-SVM)"

    tag = (
        f"{model_name} enc={cfg.encoding_name} params={cfg.encoding_params} "
        f"qubits={num_qubits} feats={len(cfg.feat_idx)} "
        f"C={cfg.C} lssvm_reg={getattr(cfg, 'lssvm_reg', None)} shots={cfg.shots} "
        f"center={cfg.center} psd_fix={cfg.psd_fix} nystrom_m={cfg.nystrom_m}"
    )

    if not return_result:
        return acc, cm, tag

    circuits = None
    if capture_circuits:
        try:
            if len(Xtr) >= 2:
                circuits = [_build_overlap_circuit(enc, Xtr[0], Xtr[1])]
            else:
                circuits = [_build_overlap_circuit(enc, Xtr[0], Xtr[0])]
        except Exception:
            circuits = None

    result = RunResult(
        y_true=yte,
        y_pred=yhat,
        y_prob=y_prob,
        K_train=(Ktr if capture_artifacts else None),
        K_test=(Kte if capture_artifacts else None),
        y_train=(ytr if capture_artifacts else None),
        circuits=circuits,
        model_name=model_name,
        encoding_name=cfg.encoding_name,
        encoding_params=cfg.encoding_params,
        feat_idx=cfg.feat_idx,
        num_params=None,
        shots=cfg.shots,
        evals=int(evals),
        timings=timings,
        extras={
            "tag": tag,
            "confusion_matrix": cm,
            "num_qubits": num_qubits,
            "variant": variant,
            "lssvm_reg": float(getattr(cfg, "lssvm_reg", 1e-3)),
        },
    )
    return acc, cm, tag, result


def run_qsvm_cv(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: QSVMConfig,
    n_splits: int = 5,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Cross-validation QSVM (accuracy mean/std). For scalability, set cfg.max_train and/or cfg.nystrom_m.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    X = X_raw[:, cfg.feat_idx]
    scaler = MinMaxScaler(feature_range=(0.0, 2 * np.pi))
    X = scaler.fit_transform(X)

    enc, _, _ = build_encoding(
        cfg.encoding_name,
        n_features=X.shape[1],
        params=cfg.encoding_params,
        X_fit=X,
        y_fit=y,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        if cfg.max_train is not None:
            Xtr, ytr = _stratified_subsample(Xtr, ytr, int(cfg.max_train), seed=seed + fold)

        Ktr = kernel_matrix(enc, Xtr, shots=cfg.shots)
        if cfg.psd_fix and cfg.shots is not None:
            Ktr = nearest_psd(Ktr)

        Kva = kernel_matrix(enc, Xva, Y=Xtr, shots=cfg.shots)
        if cfg.center:
            Ktr, Kva = center_train_test(Ktr, Kva)

        clf = SVC(kernel="precomputed", C=float(cfg.C))
        clf.fit(Ktr, ytr)
        yhat = clf.predict(Kva)
        accs.append(float((yhat == yva).mean()))

    return float(np.mean(accs)), float(np.std(accs))


def run_qsvm_combined(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfgs: List[QSVMConfig],
    test_size: float = 0.30,
    seed: int = 42,
    *,
    return_result: bool = False,
    capture_artifacts: bool = False,
    capture_circuits: bool = False,
):
    """
    Combine multiple encodings by summing their kernels, then train one SVM.

    Constraints
    ----------
    - All cfgs must share feat_idx, C, shots, center, psd_fix (only encoding differs).
    - For scalability, keep train size small; combined kernels multiply cost by #encodings.
    """
    assert len(cfgs) >= 2, "need >=2 encodings"
    base = cfgs[0]
    for c in cfgs[1:]:
        if not (
            c.feat_idx == base.feat_idx
            and float(c.C) == float(base.C)
            and c.shots == base.shots
            and bool(c.center) == bool(base.center)
            and bool(c.psd_fix) == bool(base.psd_fix)
        ):
            raise ValueError("All cfgs must match except encoding_name/encoding_params")

    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    Xsel = X_raw[:, base.feat_idx]
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xsel, y, test_size=test_size, random_state=seed, stratify=y
    )

    Xtr, Xte, scaler = _scale_features(Xtr_raw, Xte_raw)

    # Build all encodings
    encs = []
    num_qubits = None
    for c in cfgs:
        enc, nq, _ = build_encoding(
            c.encoding_name,
            n_features=Xtr.shape[1],
            params=c.encoding_params,
            X_fit=Xtr,
            y_fit=ytr,
        )
        encs.append(enc)
        num_qubits = nq if num_qubits is None else num_qubits

    timings = {}
    t0 = time.perf_counter()
    Ktr = None
    for enc in encs:
        Km = kernel_matrix(enc, Xtr, shots=base.shots)
        Ktr = Km if Ktr is None else (Ktr + Km)
    timings["kernel_train_sum"] = time.perf_counter() - t0

    if base.psd_fix and base.shots is not None:
        t_fix = time.perf_counter()
        Ktr = nearest_psd(Ktr)
        timings["kernel_psd_fix"] = time.perf_counter() - t_fix

    t0 = time.perf_counter()
    Kte = None
    for enc in encs:
        Km = kernel_matrix(enc, Xte, Y=Xtr, shots=base.shots)
        Kte = Km if Kte is None else (Kte + Km)
    timings["kernel_test_sum"] = time.perf_counter() - t0

    if base.center:
        t0 = time.perf_counter()
        Ktr, Kte = center_train_test(Ktr, Kte)
        timings["kernel_center"] = time.perf_counter() - t0

    clf = SVC(kernel="precomputed", C=float(base.C))
    t0 = time.perf_counter()
    clf.fit(Ktr, ytr)
    timings["svm_fit"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    yhat = clf.predict(Kte)
    timings["svm_predict"] = time.perf_counter() - t0

    acc = float((yhat == yte).mean())
    cm = confusion_matrix(yte, yhat)
    tag = " + ".join([f"{c.encoding_name}{c.encoding_params}" for c in cfgs])

    if not return_result:
        return acc, cm, tag

    # Eval accounting across encodings (pairwise kernel evaluations)
    n_tr, n_te = len(Xtr), len(Xte)
    evals_per_enc = (n_tr * (n_tr + 1) // 2) + (n_te * n_tr)
    total_evals = int(evals_per_enc * len(cfgs))

    circuits = None
    if capture_circuits:
        try:
            circuits = [_build_overlap_circuit(encs[0], Xtr[0], Xtr[min(1, len(Xtr)-1)])]
        except Exception:
            circuits = None

    result = RunResult(
        y_true=yte,
        y_pred=yhat,
        y_prob=None,
        K_train=(Ktr if capture_artifacts else None),
        K_test=(Kte if capture_artifacts else None),
        y_train=(ytr if capture_artifacts else None),
        circuits=circuits,
        model_name="QSVM-Combined",
        encoding_name="+".join([c.encoding_name for c in cfgs]),
        encoding_params={"configs": [c.encoding_params for c in cfgs]},
        feat_idx=base.feat_idx,
        num_params=None,
        shots=base.shots,
        evals=total_evals,
        timings=timings,
        extras={"tag": tag, "num_encodings": len(cfgs), "confusion_matrix": cm, "num_qubits": num_qubits},
    )
    return acc, cm, tag, result
