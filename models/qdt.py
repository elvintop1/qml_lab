from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from qiskit.quantum_info import Statevector

from ..encodings.helpers import build_encoding
from .protocols import RunResult


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


def _precompute_bit_and_parity_tables(num_qubits: int):
    """Precompute lookup tables over computational basis indices.

    These tables allow fast extraction of:
      - single-qubit measurement probabilities P(bit_i=1)
      - two-qubit parity probabilities P(bit_i XOR bit_j = 1)
      - global parity probability P(popcount(bits) mod 2 = 1)
    from a probability vector over basis states.

    Notes
    -----
    We use little-endian qubit indexing to match Qiskit's statevector convention used elsewhere
    in this repository.
    """
    n = int(num_qubits)
    dim = int(1 << n)
    idx = np.arange(dim, dtype=np.uint32)
    bitpos = np.arange(n, dtype=np.uint32)
    bits = ((idx[:, None] >> bitpos[None, :]) & 1).astype(np.uint8)

    # Global parity (odd popcount)
    parity_all = (bits.sum(axis=1) & 1).astype(np.uint8)

    # Pair list + parity table
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if pairs:
        i_idx = np.asarray([i for i, _ in pairs], dtype=np.int64)
        j_idx = np.asarray([j for _, j in pairs], dtype=np.int64)
        parity_pairs = (bits[:, i_idx] ^ bits[:, j_idx]).astype(np.uint8)
    else:
        parity_pairs = np.zeros((dim, 0), dtype=np.uint8)

    return bits, parity_pairs, parity_all, pairs


def _extract_axis_features_from_probs(
    probs: np.ndarray, bits_table: np.ndarray, *, feature_mode: str = "prob"
) -> np.ndarray:
    """Axis-aligned QDT features.

    The original implementation returned only hard bits, which discards how
    confident a qubit measurement is.  ``feature_mode='prob'`` keeps continuous
    probabilities P(bit_i=1), giving the downstream decision tree real margins.
    Set ``feature_mode='hard'`` to recover the earlier thesis-style ablation.
    """
    p1 = np.asarray(probs @ bits_table, dtype=float)
    mode = str(feature_mode or "prob").lower().strip()
    if mode in {"hard", "binary", "bit", "bits"}:
        return (p1 >= 0.5).astype(np.int8)
    if mode in {"both", "hybrid", "prob+hard"}:
        return np.concatenate([p1, (p1 >= 0.5).astype(float)], axis=1)
    return p1


def _extract_parity_features_from_probs(
    probs: np.ndarray, parity_table: np.ndarray, *, feature_mode: str = "prob"
) -> np.ndarray:
    """Entangled QDT features from pair-parity probabilities.

    Continuous parity probabilities are much less brittle than thresholded XOR
    bits for shallow feature maps; the result table showed HA-SAGE-CMTSD entangled
    QDT was hurt by the hard-bit bottleneck.
    """
    p_odd = np.asarray(probs @ parity_table, dtype=float)
    mode = str(feature_mode or "prob").lower().strip()
    if mode in {"hard", "binary", "bit", "bits"}:
        return (p_odd >= 0.5).astype(np.int8)
    if mode in {"both", "hybrid", "prob+hard"}:
        return np.concatenate([p_odd, (p_odd >= 0.5).astype(float)], axis=1)
    return p_odd


def _extract_cmtsd_margin_features(enc, X: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract C-MTSD latent features for the HA-SAGE-CMTSD QDT adapter.

    The earlier version returned only hard threshold bits and XOR parities.
    That threw away margin magnitude, so QDT could not learn refined thresholds
    even though the encoding had already computed them.  This version keeps
    continuous margins and bounded z-values, then appends binary bits and pair
    interactions on the learned hardware edges when available.  DecisionTreeClassifier
    can split on continuous features directly.
    """
    if not hasattr(enc, "_project"):
        raise ValueError("The cmtsd_margin QDT variant requires an encoding with a _project method, e.g. ha_sage or ha_sage_cmtsd.")
    X = np.asarray(X, dtype=float)

    U_rows = []
    M_rows = []
    Z_rows = []
    B_rows = []
    for row in X:
        out = enc._project(row)
        if isinstance(out, tuple) and len(out) == 3:
            u, margin, z = out
        elif isinstance(out, tuple) and len(out) == 2:
            u, z = out
            margin = z
        else:
            raise ValueError("Encoding _project must return (u, margin, z) or (u, z).")
        u = np.asarray(u, dtype=float)
        margin = np.asarray(margin, dtype=float)
        z = np.asarray(z, dtype=float)
        U_rows.append(u)
        M_rows.append(margin)
        Z_rows.append(z)
        B_rows.append((margin >= 0.0).astype(np.int8))

    U = np.asarray(U_rows, dtype=float)
    M = np.asarray(M_rows, dtype=float)
    Z = np.asarray(Z_rows, dtype=float)
    B = np.asarray(B_rows, dtype=np.int8)
    n = B.shape[1]
    selected_edges = getattr(enc, "edges", None)
    if selected_edges:
        pairs = [
            (int(i), int(j))
            for i, j in selected_edges
            if 0 <= int(i) < n and 0 <= int(j) < n and int(i) != int(j)
        ]
        pairs = sorted(set((min(i, j), max(i, j)) for i, j in pairs))
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    feats = [M, Z, B.astype(float)]
    if pairs:
        parity = np.column_stack([(B[:, i] ^ B[:, j]).astype(float) for i, j in pairs])
        z_prod = np.column_stack([(Z[:, i] * Z[:, j]).astype(float) for i, j in pairs])
        z_diff = np.column_stack([np.abs(Z[:, i] - Z[:, j]).astype(float) for i, j in pairs])
        u_diff = np.column_stack([np.abs(U[:, i] - U[:, j]).astype(float) for i, j in pairs])
        feats.extend([parity, z_prod, z_diff, u_diff])
    return np.concatenate(feats, axis=1), pairs


def compute_root_information_gain(tree_clf: DecisionTreeClassifier) -> float:
    """
    Information gain at the root node using impurity decrease.
    For gini criterion this matches typical IG definition.
    """
    t = tree_clf.tree_
    left = t.children_left[0]
    right = t.children_right[0]
    if left == right or left == -1 or right == -1:
        return 0.0

    N = t.n_node_samples[0]
    IG_root = t.impurity[0]

    N0 = t.n_node_samples[left]
    N1 = t.n_node_samples[right]
    IG0 = t.impurity[left]
    IG1 = t.impurity[right]

    delta = IG_root - (N0 / N) * IG0 - (N1 / N) * IG1
    return float(delta)



def _fit_decision_tree_auto(
    Xq_tr: np.ndarray,
    ytr: np.ndarray,
    *,
    criterion: str,
    max_depth: Optional[int],
    min_samples_leaf: int,
    seed: int,
    default_depth: int,
):
    """Fit a decision tree with a small validation search when depth is None.

    On the uploaded benchmark, QDT variants were often underwhelming not because
    the encoding had no useful signal, but because hard features plus a fixed
    tree depth made the split model brittle.  This helper chooses a conservative
    depth/leaf pair on the train split and then refits on all available training
    samples.  It uses balanced accuracy, matching the reported metric.
    """
    Xq_tr = np.asarray(Xq_tr, dtype=float)
    ytr = np.asarray(ytr)
    if max_depth is not None or len(Xq_tr) < 12 or np.unique(ytr).size < 2:
        depth = int(max_depth) if max_depth is not None else int(default_depth)
        clf = DecisionTreeClassifier(
            criterion=str(criterion),
            max_depth=depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=seed,
        )
        clf.fit(Xq_tr, ytr)
        return clf, depth, int(min_samples_leaf), {"auto_tree": False}

    candidates_depth = sorted(set([1, 2, 3, max(1, min(int(default_depth), 4)), int(default_depth)]))
    candidates_leaf = sorted(set([max(1, int(min_samples_leaf)), 2, 4]))
    # Keep validation feasible for tiny balanced datasets.
    try:
        Xfit, Xval, yfit, yval = train_test_split(
            Xq_tr,
            ytr,
            test_size=0.25,
            random_state=seed,
            stratify=ytr,
        )
    except Exception:
        Xfit, Xval, yfit, yval = Xq_tr, Xq_tr, ytr, ytr

    best = None
    for depth in candidates_depth:
        for leaf in candidates_leaf:
            if len(Xfit) < 2 * leaf:
                continue
            clf = DecisionTreeClassifier(
                criterion=str(criterion),
                max_depth=int(depth),
                min_samples_leaf=int(leaf),
                random_state=seed,
            )
            clf.fit(Xfit, yfit)
            pred = clf.predict(Xval)
            score = float(balanced_accuracy_score(yval, pred))
            # Tie-break toward shallower and smoother trees for stability.
            key = (score, -int(depth), -int(leaf))
            if best is None or key > best[0]:
                best = (key, int(depth), int(leaf), score)

    if best is None:
        depth = int(default_depth)
        leaf = int(min_samples_leaf)
        val_score = None
    else:
        _, depth, leaf, val_score = best

    clf = DecisionTreeClassifier(
        criterion=str(criterion),
        max_depth=int(depth),
        min_samples_leaf=int(leaf),
        random_state=seed,
    )
    clf.fit(Xq_tr, ytr)
    return clf, int(depth), int(leaf), {"auto_tree": True, "auto_tree_val_balanced_accuracy": val_score}


@dataclass
class QDTConfig:
    encoding_name: str
    encoding_params: Dict
    feat_idx: List[int]

    # --- Variants (Phase-1 thesis compatibility) ---
    # "axis": Axis-aligned QDT (Type-1) using single-qubit Z_i measurements
    # "entangled": Entangled QDT (Type-2) using two-qubit parities Z_i Z_j
    # "stump": Full-quantum stump using a global parity ancilla + Ry(theta)
    # "cmtsd_margin": HA-SAGE-CMTSD threshold-margin QDT adapter for fast ablation
    variant: str = "axis"

    # Hybrid/classical tree hyperparameters (axis/entangled)
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    criterion: str = "gini"
    feature_mode: str = "prob"  # prob | hard | both

    # Quantum execution (used for reporting / QPU estimates)
    shots: Optional[int] = None

    # Full-quantum stump hyperparameters
    stump_theta_grid: int = 21
    stump_theta_min: float = 0.0
    stump_theta_max: float = float(np.pi)
    stump_val_split: float = 0.2

    max_train: Optional[int] = None
    max_test: Optional[int] = None


def run_qdt_train_test(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: QDTConfig,
    test_size: float = 0.30,
    seed: int = 42,
    *,
    return_result: bool = False,
    capture_circuits: bool = False,
):
    """
    Quantum Decision Tree variants (thesis Phase-1 compatibility):

    - Axis-aligned QDT (Type-1): extract single-qubit Z_i outcomes (most-likely) as binary features
      and train a classical decision tree.
    - Entangled QDT (Type-2): extract two-qubit parities (Z_i Z_j) as binary features and train a
      classical decision tree.
    - Full-quantum stump: compute global parity onto an ancilla, apply Ry(θ), and use the ancilla
      measurement as the prediction (θ tuned on a validation split).

    Notes
    -----
    We use *statevector probabilities* to extract the (most-likely) measurement outcomes on a simulator.
    For QPU estimates, the returned RunResult includes gate/circuit stats and a hardware-oriented eval count.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    # -----------------------------
    # Variant selection
    # -----------------------------
    vraw = str(getattr(cfg, "variant", "axis") or "axis").lower().strip()
    vkey = vraw.replace("-", "").replace("_", "").replace(" ", "")
    if vkey in {"axis", "axisaligned", "type1", "zi"}:
        variant = "axis"
    elif vkey in {"entangled", "parity", "type2", "zizj"}:
        variant = "entangled"
    elif vkey in {"stump", "fullquantumstump", "fullquantum"}:
        variant = "stump"
    elif vkey in {"cmtsd", "cmtsdmargin", "latentmargin", "threshold", "thresholdmargin"}:
        variant = "cmtsd_margin"
    else:
        raise ValueError(
            f"Unknown QDT variant: {cfg.variant!r}. Use 'axis', 'entangled', 'stump', or 'cmtsd_margin'."
        )

    # -----------------------------
    # Split train/test
    # -----------------------------
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
    bits_table, parity_pairs, parity_all, pairs = _precompute_bit_and_parity_tables(num_qubits)

    timings: Dict[str, float] = {}

    # Helper to build a simple "encode + measure all" circuit for gate stats
    def _capture_measure_all(name: str):
        from qiskit import QuantumCircuit

        base = enc.build(Xtr[0] if len(Xtr) else Xte[0])
        qc = QuantumCircuit(base.num_qubits, name=name)
        qc.compose(base, inplace=True)
        qc.measure_all()
        return qc

    # Helper to build the full-quantum stump circuit
    def _capture_stump(theta: float):
        from qiskit import QuantumCircuit

        base = enc.build(Xtr[0] if len(Xtr) else Xte[0])
        n = base.num_qubits
        anc = n
        qc = QuantumCircuit(n + 1, 1, name="QDT_Stump")
        qc.compose(base, range(n), inplace=True)
        for q in range(n):
            qc.cx(q, anc)
        qc.ry(float(theta), anc)
        qc.measure(anc, 0)
        return qc

    # -----------------------------
    # HA-SAGE-CMTSD C-MTSD margin variant: learned thresholds -> tree
    # -----------------------------
    if variant == "cmtsd_margin":
        t0 = time.perf_counter()
        Xq_tr, cmtsd_pairs = _extract_cmtsd_margin_features(enc, Xtr)
        Xq_te, _ = _extract_cmtsd_margin_features(enc, Xte)
        timings["extract_cmtsd_margin_features"] = time.perf_counter() - t0

        default_depth = max(1, int(num_qubits))
        t0 = time.perf_counter()
        clf, depth, leaf, auto_info = _fit_decision_tree_auto(
            Xq_tr,
            ytr,
            criterion=str(cfg.criterion),
            max_depth=cfg.max_depth,
            min_samples_leaf=int(getattr(cfg, "min_samples_leaf", 1)),
            seed=int(seed),
            default_depth=default_depth,
        )
        timings["dt_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        yhat = clf.predict(Xq_te)
        timings["dt_predict"] = time.perf_counter() - t0

        acc = float(accuracy_score(yte, yhat))
        cm = confusion_matrix(yte, yhat)
        uniq = set(np.unique(yte).tolist()) | set(np.unique(yhat).tolist())
        avg = "binary" if uniq <= {0, 1} else "macro"
        f1 = float(f1_score(yte, yhat, average=avg, zero_division=0))
        info_gain = compute_root_information_gain(clf)

        model_name = "QDT (C-MTSDMargin)"
        total_qubits = int(num_qubits)
        feat_dim = int(Xq_tr.shape[1])
        tag = f"{model_name} enc={cfg.encoding_name} qubits={total_qubits} depth={depth} feat_dim={feat_dim}"
        if not return_result:
            return acc, cm, tag

        circuits = None
        if capture_circuits:
            try:
                circuits = [_capture_measure_all("QDT_CMTSD_MeasureAll")]
            except Exception:
                circuits = None

        evals = int(len(Xtr) + len(Xte))
        result = RunResult(
            y_true=yte,
            y_pred=yhat,
            y_prob=None,
            K_train=None,
            K_test=None,
            y_train=None,
            circuits=circuits,
            model_name=model_name,
            encoding_name=cfg.encoding_name,
            encoding_params=cfg.encoding_params,
            feat_idx=cfg.feat_idx,
            num_params=None,
            shots=getattr(cfg, "shots", None),
            evals=evals,
            timings=timings,
            extras={
                "tag": tag,
                "variant": variant,
                "confusion_matrix": cm,
                "f1": f1,
                "info_gain_root": info_gain,
                "num_qubits": total_qubits,
                "encoding_qubits": int(num_qubits),
                "dt_depth": depth,
                "dt_min_samples_leaf": leaf,
                **auto_info,
                "feature_dim": feat_dim,
                "n_pairs": int(len(cmtsd_pairs)),
                "pairs": cmtsd_pairs,
            },
        )
        return acc, cm, tag, result

    # -----------------------------
    # Axis-aligned and entangled variants: feature extraction -> classical tree
    # -----------------------------
    if variant in {"axis", "entangled"}:
        # Encode to statevector probabilities
        t0 = time.perf_counter()
        probs_tr = np.asarray([
            np.abs(Statevector.from_instruction(enc.build(x)).data) ** 2 for x in Xtr
        ], dtype=float)
        probs_te = np.asarray([
            np.abs(Statevector.from_instruction(enc.build(x)).data) ** 2 for x in Xte
        ], dtype=float)
        timings["encode_statevectors"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feature_mode = str(getattr(cfg, "feature_mode", "prob") or "prob")
        if variant == "axis":
            Xq_tr = _extract_axis_features_from_probs(probs_tr, bits_table, feature_mode=feature_mode)
            Xq_te = _extract_axis_features_from_probs(probs_te, bits_table, feature_mode=feature_mode)
            timings["extract_axis_features"] = time.perf_counter() - t0
            model_name = "QDT (AxisAligned)"
            total_qubits = int(num_qubits)
            feat_dim = int(Xq_tr.shape[1])
        else:
            if int(num_qubits) < 2:
                raise ValueError("Entangled QDT requires num_qubits >= 2.")
            Xq_tr = _extract_parity_features_from_probs(probs_tr, parity_pairs, feature_mode=feature_mode)
            Xq_te = _extract_parity_features_from_probs(probs_te, parity_pairs, feature_mode=feature_mode)
            timings["extract_parity_features"] = time.perf_counter() - t0
            model_name = "QDT (EntangledParity)"
            total_qubits = int(num_qubits)
            feat_dim = int(Xq_tr.shape[1])

        default_depth = max(1, int(num_qubits))
        t0 = time.perf_counter()
        clf, depth, leaf, auto_info = _fit_decision_tree_auto(
            Xq_tr,
            ytr,
            criterion=str(cfg.criterion),
            max_depth=cfg.max_depth,
            min_samples_leaf=int(getattr(cfg, "min_samples_leaf", 1)),
            seed=int(seed),
            default_depth=default_depth,
        )
        timings["dt_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        yhat = clf.predict(Xq_te)
        timings["dt_predict"] = time.perf_counter() - t0

        acc = float(accuracy_score(yte, yhat))
        cm = confusion_matrix(yte, yhat)
        uniq = set(np.unique(yte).tolist()) | set(np.unique(yhat).tolist())
        avg = "binary" if uniq <= {0, 1} else "macro"
        f1 = float(f1_score(yte, yhat, average=avg, zero_division=0))
        info_gain = compute_root_information_gain(clf)

        tag = f"{model_name} enc={cfg.encoding_name} qubits={total_qubits} depth={depth} feat_dim={feat_dim}"
        if not return_result:
            return acc, cm, tag

        circuits = None
        if capture_circuits:
            try:
                circuits = [_capture_measure_all("QDT_MeasureAll")]
            except Exception:
                circuits = None

        evals = int(len(Xtr) + len(Xte))
        result = RunResult(
            y_true=yte,
            y_pred=yhat,
            y_prob=None,
            K_train=None,
            K_test=None,
            y_train=None,
            circuits=circuits,
            model_name=model_name,
            encoding_name=cfg.encoding_name,
            encoding_params=cfg.encoding_params,
            feat_idx=cfg.feat_idx,
            num_params=None,
            shots=getattr(cfg, "shots", None),
            evals=evals,
            timings=timings,
            extras={
                "tag": tag,
                "variant": variant,
                "confusion_matrix": cm,
                "f1": f1,
                "info_gain_root": info_gain,
                "num_qubits": total_qubits,
                "encoding_qubits": int(num_qubits),
                "dt_depth": depth,
                "dt_min_samples_leaf": leaf,
                "feature_mode": feature_mode,
                **auto_info,
                "feature_dim": feat_dim,
                "n_pairs": int(len(pairs)),
                "pairs": pairs,
            },
        )
        return acc, cm, tag, result

    # -----------------------------
    # Full-quantum stump variant
    # -----------------------------
    labs = set(np.unique(ytr).tolist()) | set(np.unique(yte).tolist())
    if not (labs <= {0, 1}):
        raise ValueError("Full-quantum stump currently supports only binary labels {0,1}.")

    # Validation split for tuning θ
    val_split = float(getattr(cfg, "stump_val_split", 0.2))
    val_split = min(max(val_split, 0.0), 0.9)
    if val_split <= 0.0:
        Xval, yval = Xtr, ytr
    else:
        _, Xval, _, yval = train_test_split(
            Xtr, ytr, test_size=val_split, random_state=seed, stratify=ytr
        )

    # Encode only validation + test samples
    t0 = time.perf_counter()
    probs_val = np.asarray([
        np.abs(Statevector.from_instruction(enc.build(x)).data) ** 2 for x in Xval
    ], dtype=float)
    probs_te = np.asarray([
        np.abs(Statevector.from_instruction(enc.build(x)).data) ** 2 for x in Xte
    ], dtype=float)
    timings["encode_statevectors"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_odd_val = (probs_val @ parity_all).astype(float)
    p_odd_te = (probs_te @ parity_all).astype(float)
    timings["parity_probabilities"] = time.perf_counter() - t0

    # Grid search θ
    t0 = time.perf_counter()
    n_grid = int(getattr(cfg, "stump_theta_grid", 21))
    n_grid = max(n_grid, 2)
    tmin = float(getattr(cfg, "stump_theta_min", 0.0))
    tmax = float(getattr(cfg, "stump_theta_max", float(np.pi)))
    thetas = np.linspace(tmin, tmax, num=n_grid, dtype=float)

    best_theta = float(thetas[0])
    best_acc = -1.0
    for th in thetas:
        p1_val = 0.5 + (p_odd_val - 0.5) * float(np.cos(th))
        yhat_val = (p1_val >= 0.5).astype(int)
        a = float(accuracy_score(yval, yhat_val))
        if a > best_acc:
            best_acc = a
            best_theta = float(th)
    timings["theta_tuning"] = time.perf_counter() - t0

    # Predict on test
    t0 = time.perf_counter()
    p1_te = 0.5 + (p_odd_te - 0.5) * float(np.cos(best_theta))
    yhat = (p1_te >= 0.5).astype(int)
    y_prob = np.vstack([1.0 - p1_te, p1_te]).T
    timings["stump_predict"] = time.perf_counter() - t0

    acc = float(accuracy_score(yte, yhat))
    cm = confusion_matrix(yte, yhat)
    f1 = float(f1_score(yte, yhat, average="binary", zero_division=0))

    model_name = "QDT (FullQuantumStump)"
    total_qubits = int(num_qubits + 1)
    tag = (
        f"{model_name} enc={cfg.encoding_name} qubits={total_qubits} "
        f"theta={best_theta:.6f} val_acc={best_acc:.4f}"
    )

    if not return_result:
        return acc, cm, tag

    circuits = None
    if capture_circuits:
        try:
            circuits = [_capture_stump(best_theta)]
        except Exception:
            circuits = None

    evals = int(len(Xval) + len(Xte))
    result = RunResult(
        y_true=yte,
        y_pred=yhat,
        y_prob=y_prob,
        K_train=None,
        K_test=None,
        y_train=None,
        circuits=circuits,
        model_name=model_name,
        encoding_name=cfg.encoding_name,
        encoding_params=cfg.encoding_params,
        feat_idx=cfg.feat_idx,
        num_params=None,
        shots=getattr(cfg, "shots", None),
        evals=evals,
        timings=timings,
        extras={
            "tag": tag,
            "variant": variant,
            "confusion_matrix": cm,
            "f1": f1,
            "num_qubits": total_qubits,
            "encoding_qubits": int(num_qubits),
            "theta": float(best_theta),
            "theta_grid": int(n_grid),
            "theta_min": float(tmin),
            "theta_max": float(tmax),
            "val_split": float(val_split),
            "n_val": int(len(Xval)),
        },
    )
    return acc, cm, tag, result
