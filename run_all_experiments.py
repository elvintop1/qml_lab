#!/usr/bin/env python3
"""Unified experiment runner for all QML models and all encoding families.

This is the only experiment entry point in the cleaned repository.

Common usage from the repository root
-------------------------------------

Run every supported classifier with every canonical encoding on the repo dataset::

    python run_all_experiments.py \
        --models all \
        --encodings all \
        --datasets repo_balanced_42 \
        --max-train 64 \
        --max-test 32 \
        --epochs 5 \
        --out results/all_models_all_encodings.csv

Reproduce the broader feature sweep used in the review CSV::

    python run_all_experiments.py \
        --datasets all \
        --models all \
        --encodings all \
        --max-train 64 \
        --max-test 32 \
        --epochs 5 \
        --out results/full_feature_sweep.csv

The runner records failures as rows in the output CSV instead of stopping the
entire sweep.  Use ``--fail-fast`` when debugging one combination.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ---------------------------------------------------------------------------
# Canonical sweep definitions
# ---------------------------------------------------------------------------

# One canonical name per model implementation/variant.
ALL_MODELS: List[str] = [
    "qnn",
    "qcnn",
    "qsvm",
    "qknn",
    "qdt_axis",
    "qdt_entangled",
    "qdt_stump",
    "qdt_cmtsd",
]

# One canonical name per encoding implementation.  Aliases such as ``iqp`` for
# ``havlicek`` are deliberately not repeated because they would benchmark the
# same code twice.
ALL_ENCODINGS: List[str] = [
    "ha_sage",
    "ha_sage_cmtsd",
    "hardware_aware",
    "amplitude",
    "histogram",
    "sparse_amplitude",
    "angle",
    "denseangle",
    "reuploading",
    "havlicek",
    "hamiltonian",
    "trainable_kernel",
    "zzprod",
    "basis",
    "integer",
    "onehot",
]

# The sklearn feature sweep names are loaded lazily so that --list works even
# on machines without optional quantum dependencies.
FEATURE_SWEEP_DEFAULT: List[str] = [
    "iris_4",
    "synthetic_8",
    "wine_13",
    "synthetic_16",
    "breast_cancer_30",
    "digits_32var",
    "synthetic_32",
]

ALL_DATASETS: List[str] = ["repo_balanced_42", *FEATURE_SWEEP_DEFAULT]


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------


def _ensure_qml_lab_importable() -> Path:
    """Make the current extracted folder importable as ``qml_lab``.

    The repository ``__init__.py`` imports all encodings and models, which in
    turn import Qiskit.  We deliberately avoid executing ``__init__.py`` here so
    that commands such as ``python run_all_experiments.py --list`` still work on
    machines where Qiskit is not installed yet.  Submodules are loaded lazily
    from ``module.__path__`` when an actual experiment is run.
    """

    repo_root = Path(__file__).resolve().parent
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    existing = sys.modules.get("qml_lab")
    if existing is not None and hasattr(existing, "__path__"):
        return repo_root

    module = types.ModuleType("qml_lab")
    module.__file__ = str(repo_root / "__init__.py")
    module.__path__ = [str(repo_root)]
    module.__package__ = "qml_lab"
    sys.modules["qml_lab"] = module
    return repo_root


REPO_ROOT = _ensure_qml_lab_importable()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _canon(text: str) -> str:
    return str(text).lower().strip().replace("-", "_").replace(" ", "").replace("_", "")


def _parse_csv_list(text: str | Iterable[str], *, all_values: List[str]) -> List[str]:
    if isinstance(text, str):
        raw = [p.strip() for p in text.split(",") if p.strip()]
    else:
        raw = [str(p).strip() for p in text if str(p).strip()]
    if not raw or any(_canon(p) in {"all", "*"} for p in raw):
        return list(all_values)
    out: List[str] = []
    seen = set()
    for item in raw:
        key = _canon(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _encode_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    if np.issubdtype(y.dtype, np.number):
        classes = np.unique(y)
        if set(classes.tolist()) <= {0, 1}:
            return y.astype(int)
    return LabelEncoder().fit_transform(y).astype(int)


def _read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path).values.astype(float)


def _read_csv_vector(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return (df.iloc[:, 0].values if df.shape[1] == 1 else df.values.ravel())


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    labels, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(labels, counts)}


def _balance_classes(X: np.ndarray, y: np.ndarray, *, seed: int, enabled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Down-sample every class to the minority count for fair balanced accuracy."""

    X = np.asarray(X)
    y = np.asarray(y).astype(int).ravel()
    if not enabled:
        return X, y
    labels, counts = np.unique(y, return_counts=True)
    if labels.size <= 1:
        return X, y
    target = int(np.min(counts))
    rng = np.random.default_rng(int(seed))
    idx: List[np.ndarray] = []
    for label in labels:
        pool = np.where(y == label)[0]
        idx.append(rng.choice(pool, size=target, replace=False))
    keep = np.concatenate(idx)
    rng.shuffle(keep)
    return X[keep], y[keep]


def _stratified_sample_limit(X: np.ndarray, y: np.ndarray, limit: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).astype(int).ravel()
    if limit is None or int(limit) <= 0 or int(limit) >= len(y):
        return X, y
    labels, counts = np.unique(y, return_counts=True)
    if labels.size <= 1:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(y), size=int(limit), replace=False)
        return X[idx], y[idx]

    props = counts / counts.sum()
    target = np.maximum(1, np.floor(props * int(limit)).astype(int))
    while int(target.sum()) < int(limit):
        target[int(np.argmax(props))] += 1
    while int(target.sum()) > int(limit):
        idx = int(np.argmax(target))
        if target[idx] > 1:
            target[idx] -= 1
        else:
            break

    rng = np.random.default_rng(int(seed))
    chosen: List[np.ndarray] = []
    for label, count in zip(labels, target):
        pool = np.where(y == label)[0]
        chosen.append(rng.choice(pool, size=min(int(count), len(pool)), replace=False))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _feature_index(n_features: int, max_features: Optional[int]) -> List[int]:
    if max_features is None or int(max_features) <= 0:
        return list(range(int(n_features)))
    return list(range(min(int(max_features), int(n_features))))


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    average = "binary" if labels <= {0, 1} else "macro"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def _safe_recall(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])).astype(int)
    if int(label) not in labels:
        return float("nan")
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    pos = list(labels).index(int(label))
    return float(recalls[pos])


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": _safe_f1(y_true, y_pred),
        "recall_class_0": _safe_recall(y_true, y_pred, 0),
        "recall_class_1": _safe_recall(y_true, y_pred, 1),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_repo_balanced_42() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    d = REPO_ROOT / "datasets"
    X = np.vstack([
        _read_csv_matrix(d / "X_train_balanced_42features.csv"),
        _read_csv_matrix(d / "X_test_balanced_42features.csv"),
    ])
    y = np.concatenate([
        _read_csv_vector(d / "y_train_balanced.csv"),
        _read_csv_vector(d / "y_test_balanced.csv"),
    ])
    y = _encode_labels(y)
    meta = {"name": "repo_balanced_42", "source": "repository balanced CSV", "features": int(X.shape[1])}
    return X.astype(float), y.astype(int), meta


def _load_unbalanced_repo_csv() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    d = REPO_ROOT / "datasets"
    X = np.vstack([
        _read_csv_matrix(d / "X_train.csv"),
        _read_csv_matrix(d / "X_test.csv"),
    ])
    y = np.concatenate([
        _read_csv_vector(d / "y_train.csv"),
        _read_csv_vector(d / "y_test.csv"),
    ])
    y = _encode_labels(y)
    meta = {"name": "repo_unbalanced", "source": "repository original CSV", "features": int(X.shape[1])}
    return X.astype(float), y.astype(int), meta


def _load_single_csv(path: Path, label_column: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"label column {label_column!r} not found. Columns: {list(df.columns)}")
    y = _encode_labels(df[label_column].values)
    X = df.drop(columns=[label_column]).select_dtypes(include=[np.number]).values.astype(float)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found after dropping the label column.")
    meta = {"name": path.stem, "source": str(path), "features": int(X.shape[1])}
    return X, y, meta


def load_dataset_by_name(name: str, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    key = str(name).strip()
    canon = _canon(key)

    if canon in {"repo", "builtin", "repobalanced42"}:
        return _load_repo_balanced_42()
    if canon in {"repounbalanced", "unbalanced"}:
        return _load_unbalanced_repo_csv()

    if key.lower().startswith("csv:"):
        return _load_single_csv(Path(key.split(":", 1)[1]), args.label_column)

    # sklearn/feature-sweep datasets shipped by this repository.
    try:
        from qml_lab.datasets.feature_sweep import FEATURE_SWEEP_LOADERS, load_feature_sweep_dataset
    except Exception as exc:
        raise RuntimeError("Could not import qml_lab.datasets.feature_sweep.") from exc

    if key not in FEATURE_SWEEP_LOADERS:
        valid = ["repo_balanced_42", "repo_unbalanced", *FEATURE_SWEEP_LOADERS]
        raise ValueError(f"Unknown dataset {name!r}. Valid examples: {valid}; or use csv:/path/to/file.csv")

    X, y, meta = load_feature_sweep_dataset(key, seed=int(args.seed), limit=args.dataset_limit)
    meta = dict(meta)
    meta.setdefault("name", key)
    meta.setdefault("source", meta.get("source", key))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), meta


# ---------------------------------------------------------------------------
# Encoding configuration
# ---------------------------------------------------------------------------


def _rotational_qubits(args: argparse.Namespace, n_features: int) -> int:
    return max(1, min(int(args.qubits), int(n_features)))


def _feature_qubits(n_features: int) -> int:
    return max(1, int(n_features))


def _integer_theoretical_qubits(n_features: int, levels: int) -> int:
    del levels
    return max(1, 2 * int(n_features))


def _onehot_theoretical_qubits(n_features: int, levels: int) -> int:
    del levels
    return max(1, 4 * int(n_features))


def _is_theoretical_discrete_encoding_name(encoding: str) -> bool:
    return _canon(encoding) in {"integer", "binaryinteger", "onehot", "onehotbasis"}


def _theoretical_discrete_qubits(encoding: str, n_features: int, levels: int) -> int:
    key = _canon(encoding)
    if key in {"integer", "binaryinteger"}:
        return _integer_theoretical_qubits(n_features, levels)
    if key in {"onehot", "onehotbasis"}:
        return _onehot_theoretical_qubits(n_features, levels)
    raise ValueError(f"{encoding!r} is not a theoretical discrete encoding")


def _theoretical_total_qubits(model_key: str, encoding_qubits: int) -> int:
    if model_key == "qknn":
        return int(2 * int(encoding_qubits) + 1)
    if model_key == "qdtstump":
        return int(encoding_qubits) + 1
    return int(encoding_qubits)


def _denseangle_qubits(args: argparse.Namespace, n_features: int) -> int:
    return max(1, int(math.ceil(float(n_features) / 2.0)))


def encoding_params(name: str, args: argparse.Namespace, *, n_features: int, model_name: str = "") -> Dict[str, Any]:
    """Return clean parameters for one encoding family.

    Parameters are intentionally explicit so that other researchers can inspect
    the CSV and reproduce a circuit exactly.
    """

    key = _canon(name)
    q = _rotational_qubits(args, n_features)

    if key in {"hasage", "hardwareawarespectralalignmentbase", "hardwareawarespectralalignmentlite"}:
        return {
            "target_qubits": q,
            "layers": int(args.layers),
            "hardware_topology": str(args.hardware_topology),
            "edge_policy": str(args.edge_policy),
            "max_edges": args.max_edges,
            "seed": int(args.seed),
            "barriers": bool(args.barriers),
            "tau": float(args.tau),
            "cx_error": float(args.cx_error),
            "single_scale": min(float(args.single_scale), 1.0),
            "phase_scale": min(float(args.phase_scale), 0.45),
            "rx_scale": min(float(args.rx_scale), 0.05),
            "pair_scale": min(float(args.pair_scale), 0.55),
            "eta_product": 0.65,
            "eta_margin": 0.35,
            "axis_weight_floor": float(args.axis_weight_floor),
            "ha_sage_mode": str(args.ha_sage_mode),
            "bootstrap_rounds": int(args.bootstrap_rounds),
            "stability_weight": float(args.stability_weight),
            "risk_penalty": float(args.risk_penalty),
            "redundancy_weight": float(args.redundancy_weight),
            "post_mixer_scale": min(float(args.post_mixer_scale), 0.08),
        }

    if key in {"hasagecmtsd", "hasagecmtsdr", "rhasage", "rhasagecmtsd", "cmtsdhasage", "hardwareawarespectralalignmentcmtsd", "hasageplus", "hasagepp", "hasage++", "hardwareawarespectralalignment", "hasageplusanchor", "hasageppanchor", "hasageplusguarded", "hasageppguarded", "hasageplussageanchor", "hasageppsageanchor", "hasageplusstable", "hasageppstable", "hasageplussuperguarded", "hasageppsuperguarded"}:
        params: Dict[str, Any] = {
            "target_qubits": q,
            "layers": int(args.layers),
            "features_per_qubit": int(args.features_per_qubit),
            "hardware_topology": str(args.hardware_topology),
            "edge_policy": str(args.edge_policy),
            "max_edges": args.max_edges,
            "seed": int(args.seed),
            "barriers": bool(args.barriers),
            "cmtsd_mode": str(args.cmtsd_mode),  # v8 default: risk_guarded
            "nca_max_iter": int(args.nca_max_iter),
            "mi_weight": float(args.mi_weight),
            "relief_weight": float(args.relief_weight),
            "redundancy_weight": float(args.redundancy_weight),
            "projection_jitter": float(args.projection_jitter),
            "projection_guard_margin": float(args.projection_guard_margin),
            "kta_weight": float(args.kta_weight),
            "stability_weight": float(args.stability_weight),
            "risk_penalty": float(args.risk_penalty),
            "orthogonalize_projection": bool(args.orthogonalize_projection),
            "prototypes_per_class": int(args.prototypes_per_class),
            "use_prototypes": bool(args.use_prototypes),
            "tau": float(args.tau),
            "cx_error": float(args.cx_error),
            "single_scale": float(args.single_scale),
            "phase_scale": float(args.phase_scale),
            "rx_scale": float(args.rx_scale),
            "pair_scale": float(args.pair_scale),
            "eta_product": float(args.eta_product),
            "eta_spectral": float(args.eta_spectral),
            "eta_margin": float(args.eta_margin),
            "rff_gamma": float(args.rff_gamma),
            "axis_weight_floor": float(args.axis_weight_floor),
            "post_mixer_scale": float(args.post_mixer_scale),
        }
        model_key = _canon(model_name)
        # Historical names such as ha_sage_plus_anchor/guarded/stable are now
        # compatibility aliases only. They no longer select separate variants;
        # the unified default is the empirically stable HA-SAGE-style anchor.
        if model_key in {"qknn", "qsvm"} and bool(args.kernel_safe_cmtsd):
            params["kernel_safe"] = True
        if model_key in {"qdt", "qdtaxis", "qdtentangled", "qdtcmtsd", "qdtstump"}:
            # QDT uses probabilities or C-MTSD margins; on the full feature sweep,
            # the expressive C-MTSD phases/features overfit several
            # 64-sample splits.  Use a conservative QDT-safe profile while
            # leaving QNN/QCNN expressive.
            params["features_per_qubit"] = min(int(params["features_per_qubit"]), 2)
            params["phase_scale"] = min(float(params["phase_scale"]), 0.45)
            params["rx_scale"] = min(float(params["rx_scale"]), 0.05)
            params["pair_scale"] = min(float(params["pair_scale"]), 0.55)
            params["eta_product"] = max(float(params["eta_product"]), 0.55)
            params["eta_spectral"] = min(float(params["eta_spectral"]), 0.10)
            params["eta_margin"] = max(float(params["eta_margin"]), 0.35)
            params["post_mixer_scale"] = min(float(params["post_mixer_scale"]), 0.08)
        return {k: v for k, v in params.items() if v is not None}

    if key in {"hardwareaware", "hardwareawareflexible", "haflex", "newencoding", "hardwareawarecompact"}:
        params = {
            "target_qubits": q,
            "layers": int(args.layers),
            "hardware_topology": str(args.hardware_topology),
            "edge_policy": str(args.edge_policy),
            "max_edges": args.max_edges,
            "seed": int(args.seed),
            "barriers": bool(args.barriers),
            "feature_select": "mi_corr",
            "feature_weighting": "mi",
            "tau": float(args.tau),
            "lambda_mi": 0.25,
            "single_scale": 1.0,
            "phase_scale": 0.25,
            "pair_scale": 0.75,
        }
        return {k: v for k, v in params.items() if v is not None}

    if key == "amplitude":
        params = {"rescale": "unit"}
        if args.amplitude_fixed_qubits:
            params["num_qubits"] = q
        return params

    if key in {"histogram", "probability", "probabilityhistogram"}:
        params = {"mode": str(args.histogram_mode)}
        if args.amplitude_fixed_qubits:
            params["num_qubits"] = q
        return params

    if key in {"sparse", "sparseamplitude"}:
        params = {"rescale": "unit", "threshold": float(args.sparse_threshold)}
        if args.amplitude_fixed_qubits:
            params["num_qubits"] = q
        return params

    if key == "angle":
        return {"num_qubits": _feature_qubits(n_features), "axis": str(args.angle_axis), "alpha": float(args.angle_alpha)}

    if key == "denseangle":
        return {
            "num_qubits": _denseangle_qubits(args, n_features),
            "entangle": str(args.dense_entangle),
            "scale_theta": float(args.dense_scale_theta),
            "scale_phi": float(args.dense_scale_phi),
        }

    if key in {"reuploading", "reupload", "datareuploading"}:
        return {
            "num_qubits": _feature_qubits(n_features),
            "layers": int(args.layers),
            "axes": str(args.reupload_axes),
            "alpha": float(args.reupload_alpha),
            "entangle": str(args.reupload_entangle),
            "hadamard": bool(args.reupload_hadamard),
            "seed": int(args.seed),
        }

    if key in {"havlicek", "iqp", "zzfeaturemap"}:
        return {"num_qubits": _feature_qubits(n_features), "entangler": str(args.correlation_entangler), "pair": str(args.pair_function), "reps": int(args.layers)}

    if key in {"hamiltonian", "hamiltonianevolution"}:
        return {
            "num_qubits": _feature_qubits(n_features),
            "entangler": str(args.correlation_entangler),
            "pair": str(args.pair_function),
            "time": float(args.hamiltonian_time),
            "hadamard": bool(args.hamiltonian_hadamard),
        }

    if key in {"trainablekernel", "trainablekernelmap"}:
        return {
            "num_qubits": _feature_qubits(n_features),
            "layers": int(args.layers),
            "entangler": str(args.correlation_entangler),
            "seed": int(args.seed),
            "weight_scale": float(args.trainable_weight_scale),
            "data_scale": float(args.trainable_data_scale),
            "hadamard": bool(args.trainable_hadamard),
        }

    if key in {"zzprod", "zzproduct"}:
        return {
            "num_qubits": _feature_qubits(n_features),
            "scale": float(args.zz_scale),
            "entangler": str(args.correlation_entangler),
            "hadamard": bool(args.zz_hadamard),
            "final_hadamard": bool(args.zz_final_hadamard),
        }

    if key == "basis":
        return {"num_qubits": _feature_qubits(n_features), "threshold": str(args.basis_threshold)}

    if key in {"integer", "binaryinteger"}:
        levels = int(args.discrete_levels)
        return {
            "num_qubits": _integer_theoretical_qubits(n_features, levels),
            "levels": levels,
            "seed": int(args.seed),
            "theoretical_only": True,
        }

    if key in {"onehot", "onehotbasis"}:
        levels = int(args.discrete_levels)
        return {
            "num_qubits": _onehot_theoretical_qubits(n_features, levels),
            "levels": levels,
            "seed": int(args.seed),
            "theoretical_only": True,
        }

    return {}


# ---------------------------------------------------------------------------
# Circuit/resource helpers
# ---------------------------------------------------------------------------


def _count_multi_qubit_gates(qc: Any, arity: int | None = None) -> int:
    total = 0
    for inst in getattr(qc, "data", []):
        try:
            qargs = inst.qubits  # qiskit >= 1.0 CircuitInstruction
        except Exception:
            qargs = inst[1]      # older tuple format
        nq = len(qargs)
        if arity is None:
            if nq >= 2:
                total += 1
        elif nq == arity:
            total += 1
    return int(total)


def circuit_stats(qc: Any | None, prefix: str) -> Dict[str, Any]:
    if qc is None:
        return {
            f"{prefix}_qubits": np.nan,
            f"{prefix}_depth": np.nan,
            f"{prefix}_size": np.nan,
            f"{prefix}_two_qubit_gates": np.nan,
            f"{prefix}_multi_qubit_gates": np.nan,
            f"{prefix}_op_counts": json.dumps({}),
        }
    try:
        ops = {str(k): int(v) for k, v in qc.count_ops().items()}
    except Exception:
        ops = {}
    return {
        f"{prefix}_qubits": int(getattr(qc, "num_qubits", 0)),
        f"{prefix}_depth": int(qc.depth()) if hasattr(qc, "depth") else np.nan,
        f"{prefix}_size": int(qc.size()) if hasattr(qc, "size") else np.nan,
        f"{prefix}_two_qubit_gates": _count_multi_qubit_gates(qc, arity=2),
        f"{prefix}_multi_qubit_gates": _count_multi_qubit_gates(qc, arity=None),
        f"{prefix}_op_counts": json.dumps(ops, sort_keys=True),
    }



def _compact_encoding_summary(summary: Any) -> Dict[str, Any]:
    """Keep only lightweight fitted-encoding diagnostics for CSV output."""
    if not isinstance(summary, dict):
        return {}
    keep = {
        "algorithm",
        "projection",
        "projection_scores",
        "num_samples_fit",
        "num_raw_features",
        "num_qubits",
        "features_per_qubit",
        "num_selected_features",
        "selected_features",
        "feature_indices",
        "edges",
        "prototype_indices",
    }
    out: Dict[str, Any] = {}
    for key in keep:
        if key not in summary:
            continue
        value = summary[key]
        if key == "prototype_indices" and isinstance(value, list):
            out["num_prototypes_from_encoder"] = len(value)
            continue
        if isinstance(value, list) and len(value) > 16:
            out[key] = value[:16]
            out[f"{key}_truncated"] = len(value)
        elif isinstance(value, dict):
            out[key] = {
                str(k): (round(float(v), 6) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out

def build_encoding_stats(
    encoding_name: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: List[int],
    *,
    max_fit_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build one representative encoding circuit for resource accounting."""

    from qml_lab.encodings.helpers import build_encoding

    Xsel = np.asarray(X, dtype=float)[:, feat_idx]
    y_fit = np.asarray(y, dtype=int)
    Xsel, y_fit = _stratified_sample_limit(Xsel, y_fit, max_fit_samples, int(seed))
    scaler = MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))
    Xs = scaler.fit_transform(Xsel)
    enc, _, _ = build_encoding(
        encoding_name,
        n_features=Xs.shape[1],
        params=dict(params),
        X_fit=Xs,
        y_fit=y_fit,
    )
    qc = enc.build(Xs[0])
    stats = circuit_stats(qc, "encoding")
    stats["encoding_fit_samples"] = int(len(y_fit))
    fit_summary = getattr(enc, "cmtsd_summary", None)
    if not fit_summary:
        fit_summary = getattr(enc, "ha_sage_summary", None)
    stats["encoding_fit_summary"] = json.dumps(_compact_encoding_summary(fit_summary), sort_keys=True, default=_json_default)
    return stats


# ---------------------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------------------


def _timing_columns(timings: Dict[str, Any]) -> Dict[str, float]:
    timings = timings or {}
    def get_any(*names: str) -> float:
        return float(sum(float(timings.get(n, 0.0) or 0.0) for n in names))

    train_seconds = get_any("train", "fit", "dt_fit", "svm_fit", "lssvm_solve")
    predict_seconds = get_any("predict", "dt_predict", "knn_vote", "svm_predict", "lssvm_predict")
    kernel_seconds = get_any("kernel_total", "fidelity_statevector", "fidelity_shots", "encode_statevectors")
    return {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "kernel_seconds": kernel_seconds,
    }


def _write_csv_safely(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    last_exc: Optional[BaseException] = None
    for _ in range(5):
        try:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.25)
    if last_exc is not None:
        raise last_exc


def _is_cmtsd_encoding_name(encoding: str) -> bool:
    return _canon(encoding) in {
        "hasagecmtsd",
        "hasagecmtsdr",
        "rhasage",
        "rhasagecmtsd",
        "cmtsdhasage",
        "hardwareawarespectralalignmentcmtsd",
        "hasageplus",
        "hasagepp",
        "hasage++",
        "hardwareawarespectralalignment",
        "hasageplusanchor",
        "hasageppanchor",
        "hasageplusguarded",
        "hasageppguarded",
        "hasageplussageanchor",
        "hasageppsageanchor",
        "hasageplusstable",
        "hasageppstable",
        "hasageplussuperguarded",
        "hasageppsuperguarded",
    }


def _supports_cmtsd_margin(encoding: str) -> bool:
    return _canon(encoding) == "hasage" or _is_cmtsd_encoding_name(encoding)


def _resolved_qdt_feature_mode(model: str, encoding: str, requested: str) -> str:
    mode = str(requested or "auto").lower().strip()
    if mode != "auto":
        return mode
    if _canon(model) == "qdtentangled" and _is_cmtsd_encoding_name(encoding):
        return "both"
    return "prob"


def _run_qknn(encoding: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, feat_idx: List[int], args: argparse.Namespace):
    from qml_lab.models.qknn import QKNNConfig, run_qknn_train_test

    cfg = QKNNConfig(
        encoding_name=encoding,
        encoding_params=dict(params),
        feat_idx=feat_idx,
        k=int(args.k),
        shots=args.shots,
        vote=str(args.qknn_vote),
        max_train=args.max_train,
        max_test=args.max_test,
        use_encoding_prototypes=bool(args.use_prototypes),
        prototype_policy=str(args.qknn_prototype_policy),
        prototype_keep_ratio=float(args.qknn_prototype_keep_ratio),
        min_prototypes_per_class=int(args.qknn_min_prototypes_per_class),
        max_prototypes_per_class=args.qknn_max_prototypes_per_class,
        prototype_min_train_score=float(args.qknn_prototype_min_train_score),
        prototype_max_train_gap=float(args.qknn_prototype_max_train_gap),
    )
    return run_qknn_train_test(
        X,
        y,
        cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        return_result=True,
        capture_circuits=bool(args.capture_circuits),
    )


def _run_qdt(model: str, encoding: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, feat_idx: List[int], args: argparse.Namespace):
    from qml_lab.models.qdt import QDTConfig, run_qdt_train_test

    model_key = _canon(model)
    feature_mode = _resolved_qdt_feature_mode(model, encoding, str(args.qdt_feature_mode))
    if model_key == "qdtentangled":
        variant = "entangled"
    elif model_key == "qdtcmtsd":
        variant = "cmtsd_margin"
    elif model_key == "qdtstump":
        variant = "stump"
    else:
        variant = "axis"

    cfg = QDTConfig(
        encoding_name=encoding,
        encoding_params=dict(params),
        feat_idx=feat_idx,
        variant=variant,
        max_depth=args.qdt_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        criterion=str(args.qdt_criterion),
        feature_mode=feature_mode,
        shots=args.shots,
        stump_theta_grid=int(args.qdt_stump_theta_grid),
        max_train=args.max_train,
        max_test=args.max_test,
    )
    return run_qdt_train_test(
        X,
        y,
        cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        return_result=True,
        capture_circuits=bool(args.capture_circuits),
    )


def _run_qsvm(encoding: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, feat_idx: List[int], args: argparse.Namespace):
    from qml_lab.models.qsvm import QSVMConfig, run_qsvm_train_test

    cfg = QSVMConfig(
        encoding_name=encoding,
        encoding_params=dict(params),
        feat_idx=feat_idx,
        C=float(args.svm_c),
        shots=args.shots,
        variant=str(args.qsvm_variant),
        lssvm_reg=float(args.lssvm_reg),
        max_train=args.max_train,
        max_test=args.max_test,
        nystrom_m=args.nystrom_m,
        center=bool(args.kernel_center),
        psd_fix=bool(args.psd_fix),
    )
    return run_qsvm_train_test(
        X,
        y,
        cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        return_result=True,
        capture_artifacts=False,
        capture_circuits=bool(args.capture_circuits),
    )


def _run_qnn(encoding: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, feat_idx: List[int], args: argparse.Namespace):
    from qml_lab.models.qnn import QNNConfig, run_qnn_train_test

    cfg = QNNConfig(
        encoding_name=encoding,
        encoding_params=dict(params),
        feat_idx=feat_idx,
        depth=int(args.qnn_depth),
        readout=int(args.readout),
        shots=args.shots,
        epochs=int(args.epochs),
        optimizer=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        max_train=args.max_train,
        max_test=args.max_test,
    )
    return run_qnn_train_test(
        X,
        y,
        cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        return_result=True,
        capture_circuits=bool(args.capture_circuits),
    )


def _run_qcnn(encoding: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, feat_idx: List[int], args: argparse.Namespace):
    from qml_lab.models.qcnn import QCNNConfig, run_qcnn_train_test

    cfg = QCNNConfig(
        encoding_name=encoding,
        encoding_params=dict(params),
        feat_idx=feat_idx,
        stages=args.qcnn_stages,
        readout=int(args.readout),
        shots=args.shots,
        epochs=int(args.epochs),
        optimizer=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        max_train=args.max_train,
        max_test=args.max_test,
    )
    return run_qcnn_train_test(
        X,
        y,
        cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        return_result=True,
        capture_circuits=bool(args.capture_circuits),
    )


def run_one_combination(
    dataset_name: str,
    dataset_meta: Dict[str, Any],
    model: str,
    encoding: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: List[int],
    args: argparse.Namespace,
    encoding_stats_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    model_key = _canon(model)

    base: Dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_source": dataset_meta.get("source", dataset_name),
        "model": model,
        "encoding": encoding,
        "status": "OK",
        "samples_total": int(dataset_meta.get("samples_total_raw", len(y))),
        "samples_balanced": int(len(y)),
        "class_counts": json.dumps(_class_counts(y), sort_keys=True),
        "n_features": int(len(feat_idx)),
        "max_train": args.max_train,
        "max_test": args.max_test,
        "test_size": float(args.test_size),
        "epochs": int(args.epochs) if model_key in {"qnn", "qcnn"} else np.nan,
        "optimizer": str(args.optimizer) if model_key in {"qnn", "qcnn"} else "",
        "lr": float(args.lr) if model_key in {"qnn", "qcnn"} else np.nan,
        "batch_size": int(args.batch_size) if model_key in {"qnn", "qcnn"} else np.nan,
        "qdt_feature_mode": _resolved_qdt_feature_mode(model, encoding, str(args.qdt_feature_mode)) if model_key in {"qdt", "qdtaxis", "qdtentangled", "qdtcmtsd", "qdtstump"} else "",
        "params_json": json.dumps(params, sort_keys=True, default=_json_default),
    }

    if _is_theoretical_discrete_encoding_name(encoding):
        levels = int(params.get("levels", args.discrete_levels))
        encoding_qubits = _theoretical_discrete_qubits(encoding, len(feat_idx), levels)
        total_qubits = _theoretical_total_qubits(model_key, encoding_qubits)
        encoding_summary = {
            "algorithm": "theoretical-discrete-resource-only",
            "encoding": encoding,
            "levels": levels,
            "num_features": int(len(feat_idx)),
            "width_formula": "2*n_features" if _canon(encoding) in {"integer", "binaryinteger"} else "4*n_features",
            "num_qubits": encoding_qubits,
        }
        base.update({
            "status": "SKIPPED",
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "recall_class_0": np.nan,
            "recall_class_1": np.nan,
            "evals": np.nan,
            "train_seconds": np.nan,
            "predict_seconds": np.nan,
            "kernel_seconds": np.nan,
            "confusion_matrix": "",
            "tag": f"SKIPPED: {encoding} encoder width is {encoding_qubits} qubits; resource-only row, no circuit simulation",
            "extras_json": json.dumps({
                "skipped": True,
                "reason": "theoretical discrete encoding is too large to simulate faithfully",
                "encoding_qubits": encoding_qubits,
                "total_qubits": total_qubits,
            }, sort_keys=True, default=_json_default),
            "error": "",
            **circuit_stats(None, "encoding"),
            **circuit_stats(None, "model"),
            "encoding_qubits": encoding_qubits,
            "model_qubits": total_qubits,
            "total_qubits": total_qubits,
            "total_depth": np.nan,
            "total_size": np.nan,
            "total_two_qubit_gates": np.nan,
            "total_multi_qubit_gates": np.nan,
            "encoding_fit_samples": 0,
            "encoding_fit_summary": json.dumps(encoding_summary, sort_keys=True, default=_json_default),
        })
        return base

    if model_key == "qdtcmtsd" and not _supports_cmtsd_margin(encoding):
        base.update({
            "status": "SKIPPED",
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "recall_class_0": np.nan,
            "recall_class_1": np.nan,
            "evals": np.nan,
            "tag": "SKIPPED: qdt_cmtsd requires HA-SAGE/HA-SAGE-CMTSD latent projection",
            "extras_json": json.dumps({"skipped": True, "reason": "qdt_cmtsd requires ha_sage or ha_sage_cmtsd-compatible encoding"}),
            "error": "",
        })
        return base

    # Encoding resource stats are cached per dataset+encoding+parameter set.
    cache_key = json.dumps({"dataset": dataset_name, "encoding": encoding, "params": params, "feat_idx": feat_idx}, sort_keys=True, default=_json_default)
    if cache_key not in encoding_stats_cache:
        try:
            encoding_stats_cache[cache_key] = build_encoding_stats(
                encoding,
                params,
                X,
                y,
                feat_idx,
                max_fit_samples=args.max_train,
                seed=int(args.seed),
            )
        except Exception as exc:
            encoding_stats_cache[cache_key] = {**circuit_stats(None, "encoding"), "encoding_stats_error": repr(exc)}
    base.update(encoding_stats_cache[cache_key])

    if model_key == "qknn":
        acc, cm, tag, result = _run_qknn(encoding, params, X, y, feat_idx, args)
    elif model_key in {"qdt", "qdtaxis", "qdtentangled", "qdtcmtsd", "qdtstump"}:
        acc, cm, tag, result = _run_qdt(model, encoding, params, X, y, feat_idx, args)
    elif model_key == "qsvm":
        acc, cm, tag, result = _run_qsvm(encoding, params, X, y, feat_idx, args)
    elif model_key == "qnn":
        acc, cm, tag, result = _run_qnn(encoding, params, X, y, feat_idx, args)
    elif model_key == "qcnn":
        acc, cm, tag, result = _run_qcnn(encoding, params, X, y, feat_idx, args)
    else:
        raise ValueError(f"Unknown model {model!r}. Use --models all or one of {ALL_MODELS}.")

    metrics = _metric_block(result.y_true, result.y_pred)
    model_qc = None
    if getattr(result, "circuits", None):
        try:
            model_qc = result.circuits[0]
        except Exception:
            model_qc = None

    base.update(metrics)
    base.update(circuit_stats(model_qc, "model"))
    if model_qc is not None:
        base.update({
            "total_qubits": int(getattr(model_qc, "num_qubits", 0)),
            "total_depth": int(model_qc.depth()),
            "total_size": int(model_qc.size()),
            "total_two_qubit_gates": _count_multi_qubit_gates(model_qc, arity=2),
            "total_multi_qubit_gates": _count_multi_qubit_gates(model_qc, arity=None),
        })
    else:
        base.update({
            "total_qubits": np.nan,
            "total_depth": np.nan,
            "total_size": np.nan,
            "total_two_qubit_gates": np.nan,
            "total_multi_qubit_gates": np.nan,
        })

    extras = dict(getattr(result, "extras", {}) or {})
    extras.setdefault("confusion_matrix", np.asarray(cm).tolist())
    timings = dict(getattr(result, "timings", {}) or {})
    base.update(_timing_columns(timings))
    base.update({
        "evals": int(getattr(result, "evals", 0)),
        "confusion_matrix": json.dumps(np.asarray(cm).tolist()),
        "tag": tag,
        "extras_json": json.dumps(extras, sort_keys=True, default=_json_default),
        "error": "",
    })
    return base


def failure_row(
    dataset_name: str,
    dataset_meta: Dict[str, Any],
    model: str,
    encoding: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: List[int],
    args: argparse.Namespace,
    exc: BaseException,
) -> Dict[str, Any]:
    tb = traceback.format_exc(limit=8)
    model_key = _canon(model)
    return {
        "dataset": dataset_name,
        "dataset_source": dataset_meta.get("source", dataset_name),
        "model": model,
        "encoding": encoding,
        "status": "FAILED",
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "recall_class_0": np.nan,
        "recall_class_1": np.nan,
        "samples_total": int(dataset_meta.get("samples_total_raw", len(y))),
        "samples_balanced": int(len(y)),
        "class_counts": json.dumps(_class_counts(y), sort_keys=True),
        "n_features": int(len(feat_idx)),
        "max_train": args.max_train,
        "max_test": args.max_test,
        "test_size": float(args.test_size),
        "epochs": int(args.epochs) if model_key in {"qnn", "qcnn"} else np.nan,
        "optimizer": str(args.optimizer) if model_key in {"qnn", "qcnn"} else "",
        "lr": float(args.lr) if model_key in {"qnn", "qcnn"} else np.nan,
        "batch_size": int(args.batch_size) if model_key in {"qnn", "qcnn"} else np.nan,
        "qdt_feature_mode": _resolved_qdt_feature_mode(model, encoding, str(args.qdt_feature_mode)) if model_key in {"qdt", "qdtaxis", "qdtentangled", "qdtcmtsd", "qdtstump"} else "",
        "params_json": json.dumps(params, sort_keys=True, default=_json_default),
        "evals": np.nan,
        "train_seconds": np.nan,
        "predict_seconds": np.nan,
        "kernel_seconds": np.nan,
        "confusion_matrix": "",
        "tag": "FAILED",
        "extras_json": json.dumps({"error": repr(exc), "traceback": tb}),
        "error": repr(exc),
        **circuit_stats(None, "encoding"),
        **circuit_stats(None, "model"),
        "total_qubits": np.nan,
        "total_depth": np.nan,
        "total_size": np.nan,
        "total_two_qubit_gates": np.nan,
        "total_multi_qubit_gates": np.nan,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file runner for all QML models and all encoding families.")

    p.add_argument("--list", action="store_true", help="Print available models, encodings, and datasets, then exit.")
    p.add_argument("--datasets", type=str, default="repo_balanced_42", help="Comma list, 'all', or csv:/path/file.csv")
    p.add_argument("--label-column", type=str, default="label", help="Label column when using csv:/path/file.csv")
    p.add_argument("--dataset-limit", type=int, default=None, help="Optional stratified sample limit before train/test split")
    p.add_argument("--no-balance", action="store_true", help="Do not down-sample classes to equal counts before running models")

    p.add_argument("--models", type=str, default="all", help="Comma list or 'all'")
    p.add_argument("--encodings", type=str, default="all", help="Comma list or 'all'")
    p.add_argument("--out", type=str, default="results/all_models_all_encodings.csv")
    p.add_argument("--append", action="store_true", help="Append to an existing CSV instead of overwriting it")
    p.add_argument("--fail-fast", action="store_true", help="Raise the first failure instead of recording it in the CSV")
    p.add_argument("--capture-circuits", action=argparse.BooleanOptionalAction, default=True, help="Capture one representative circuit per run for resource columns")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--max-features", type=int, default=0, help="0 means use all dataset features")
    p.add_argument("--max-train", type=int, default=64)
    p.add_argument("--max-test", type=int, default=32)
    p.add_argument("--shots", type=int, default=None)

    # Shared encoding controls.
    p.add_argument("--qubits", type=int, default=16)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--features-per-qubit", type=int, default=4)
    p.add_argument("--hardware-topology", type=str, default="linear", choices=["linear", "ring", "star", "full", "grid2x4", "none"])
    p.add_argument("--edge-policy", type=str, default="matching", choices=["matching", "best", "parallel", "low_depth"])
    p.add_argument("--max-edges", type=int, default=None)
    p.add_argument("--barriers", action="store_true")
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--cx-error", type=float, default=0.01)

    # HA-SAGE-CMTSD / C-MTSD controls.
    p.add_argument("--ha-sage-mode", type=str, default="stability", choices=["stability", "legacy"], help="HA-SAGE selector. 'stability' enables v8 bootstrap reliability gating; 'legacy' reproduces the v5 selector.")
    p.add_argument("--bootstrap-rounds", type=int, default=12, help="Bootstrap rounds for HA-SAGE feature/threshold stability.")
    p.add_argument("--cmtsd-mode", type=str, default="risk_guarded", choices=["risk_guarded", "risk", "risk_kta", "orthogonal", "sage_anchor", "auto", "guarded", "anchor", "nca", "sparse", "superguarded", "legacy"], help="Projection mode for ha_sage_cmtsd. Default risk_guarded is the v8 risk-controlled selector.")
    p.add_argument("--projection-guard-margin", type=float, default=0.06, help="For guarded C-MTSD ablation, richer projections must beat the safe anchor by this score margin.")
    p.add_argument("--kta-weight", type=float, default=0.18, help="Weight of surrogate centered kernel-target alignment in the v8 C-MTSD selector.")
    p.add_argument("--stability-weight", type=float, default=0.20, help="Weight of validation/bootstrap stability in HA-SAGE/C-MTSD selectors.")
    p.add_argument("--risk-penalty", type=float, default=0.15, help="Penalty for unstable validation scores and redundant features.")
    p.add_argument("--orthogonalize-projection", action=argparse.BooleanOptionalAction, default=True, help="Include an orthogonalized sparse projection candidate for C-MTSD.")
    p.add_argument("--nca-max-iter", type=int, default=80)
    p.add_argument("--mi-weight", type=float, default=1.0)
    p.add_argument("--relief-weight", type=float, default=0.5)
    p.add_argument("--redundancy-weight", type=float, default=0.35)
    p.add_argument("--projection-jitter", type=float, default=0.0)
    p.add_argument("--prototypes-per-class", type=int, default=3)
    p.add_argument("--use-prototypes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rff-gamma", type=float, default=1.0)
    p.add_argument("--axis-weight-floor", type=float, default=0.25)
    p.add_argument("--kernel-safe-cmtsd", "--kernel-safe-ha-sage-plus", dest="kernel_safe_cmtsd", action=argparse.BooleanOptionalAction, default=True, help="Temper oscillatory C-MTSD phases for QSVM/QKNN. Old option name is kept as an alias.")
    p.add_argument("--single-scale", type=float, default=1.05)
    p.add_argument("--phase-scale", type=float, default=0.60)
    p.add_argument("--rx-scale", type=float, default=0.10)
    p.add_argument("--pair-scale", type=float, default=0.65)
    p.add_argument("--eta-product", type=float, default=0.45)
    p.add_argument("--eta-spectral", type=float, default=0.25)
    p.add_argument("--eta-margin", type=float, default=0.30)
    p.add_argument("--post-mixer-scale", type=float, default=0.12)

    # Non-HA encoding controls.
    p.add_argument("--amplitude-fixed-qubits", action="store_true", help="Force amplitude/histogram/sparse encodings to use --qubits instead of ceil(log2(features))")
    p.add_argument("--histogram-mode", type=str, default="abs", choices=["abs", "relu", "softmax"])
    p.add_argument("--sparse-threshold", type=float, default=1e-3)
    p.add_argument("--angle-axis", type=str, default="ry", choices=["rx", "ry", "rz"])
    p.add_argument("--angle-alpha", type=float, default=1.0)
    p.add_argument("--dense-entangle", type=str, default="ring", choices=["none", "linear", "ring", "full"])
    p.add_argument("--dense-scale-theta", type=float, default=1.0)
    p.add_argument("--dense-scale-phi", type=float, default=1.0)
    p.add_argument("--reupload-axes", type=str, default="ryrz")
    p.add_argument("--reupload-alpha", type=float, default=1.0)
    p.add_argument("--reupload-entangle", type=str, default="ring", choices=["none", "linear", "ring", "full"])
    p.add_argument("--reupload-hadamard", action="store_true")
    p.add_argument("--correlation-entangler", type=str, default="ring", choices=["none", "linear", "ring", "full"])
    p.add_argument("--pair-function", type=str, default="prod", choices=["prod", "cos", "gauss", "lin"])
    p.add_argument("--hamiltonian-time", type=float, default=1.0)
    p.add_argument("--hamiltonian-hadamard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trainable-weight-scale", type=float, default=0.3)
    p.add_argument("--trainable-data-scale", type=float, default=1.0)
    p.add_argument("--trainable-hadamard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--zz-scale", type=float, default=1.0)
    p.add_argument("--zz-hadamard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--zz-final-hadamard", action="store_true")
    p.add_argument("--basis-threshold", type=str, default="mid")
    p.add_argument("--discrete-levels", type=int, default=16)

    # Shared variational controls for QNN/QCNN.
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--optimizer", type=str, default="backprop", choices=["backprop", "adam", "autograd", "spsa"])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--readout", type=int, default=0)
    p.add_argument("--qnn-depth", type=int, default=2)
    p.add_argument("--qcnn-stages", type=int, default=None)

    # QKNN controls.
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--qknn-vote", type=str, default="distance_weighted", choices=["uniform", "distance_weighted", "fidelity_weighted"])
    p.add_argument("--qknn-prototype-policy", type=str, default="adaptive", choices=["off", "fixed", "adaptive", "safe"])
    p.add_argument("--qknn-prototype-keep-ratio", type=float, default=0.35)
    p.add_argument("--qknn-min-prototypes-per-class", type=int, default=0)
    p.add_argument("--qknn-max-prototypes-per-class", type=int, default=None)
    p.add_argument("--qknn-prototype-min-train-score", type=float, default=0.88)
    p.add_argument("--qknn-prototype-max-train-gap", type=float, default=0.08)

    # QDT controls.
    p.add_argument("--qdt-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--qdt-criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])
    p.add_argument("--qdt-feature-mode", type=str, default="auto", choices=["auto", "prob", "hard", "both"])
    p.add_argument("--qdt-stump-theta-grid", type=int, default=21)

    # QSVM controls.
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--qsvm-variant", type=str, default="kernel_overlap", choices=["kernel_overlap", "ls_svm"])
    p.add_argument("--lssvm-reg", type=float, default=1e-3)
    p.add_argument("--nystrom-m", type=int, default=None)
    p.add_argument("--kernel-center", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--psd-fix", action="store_true")

    return p


def print_available() -> None:
    print("Models:")
    for item in ALL_MODELS:
        print(f"  - {item}")
    print("\nEncodings:")
    for item in ALL_ENCODINGS:
        print(f"  - {item}")
    print("\nDatasets:")
    printed = set()
    for item in ALL_DATASETS:
        printed.add(item)
        print(f"  - {item}")
    try:
        from qml_lab.datasets.feature_sweep import FEATURE_SWEEP_LOADERS

        extras = [item for item in FEATURE_SWEEP_LOADERS if item not in printed]
        if extras:
            print("\nAdditional datasets (not included by --datasets all):")
            for item in extras:
                print(f"  - {item}")
    except Exception:
        pass
    print("  - csv:/path/to/file.csv")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.list:
        print_available()
        return 0

    models = _parse_csv_list(args.models, all_values=ALL_MODELS)
    encodings = _parse_csv_list(args.encodings, all_values=ALL_ENCODINGS)
    datasets = _parse_csv_list(args.datasets, all_values=ALL_DATASETS)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    existing_rows: List[Dict[str, Any]] = []
    if args.append and out_path.exists():
        existing_rows = pd.read_csv(out_path).to_dict("records")
    existing_keys = {
        (str(row.get("dataset")), str(row.get("model")), str(row.get("encoding")))
        for row in existing_rows
        if str(row.get("status", "")).upper() in {"OK", "SKIPPED"}
    }
    encoding_stats_cache: Dict[str, Dict[str, Any]] = {}
    t_all = time.perf_counter()

    print("Unified QML experiment runner")
    print(f"  datasets : {datasets}")
    print(f"  models   : {models}")
    print(f"  encodings: {encodings}")
    print(f"  output   : {out_path}")

    for dataset_name in datasets:
        X_raw, y_raw, meta = load_dataset_by_name(dataset_name, args)
        meta = dict(meta)
        meta["samples_total_raw"] = int(len(y_raw))
        X, y = _balance_classes(X_raw, y_raw, seed=int(args.seed), enabled=not bool(args.no_balance))
        feat_idx = _feature_index(X.shape[1], args.max_features)
        dataset_label = str(meta.get("name", dataset_name))
        print(f"\n[DATASET] {dataset_label}: samples={len(y)} features_used={len(feat_idx)} class_counts={_class_counts(y)}")

        for encoding in encodings:
            for model in models:
                run_key = (str(dataset_label), str(model), str(encoding))
                if args.append and run_key in existing_keys:
                    print(f"  [SKIP-EXISTING] dataset={dataset_label} model={model} encoding={encoding}", flush=True)
                    continue
                params = encoding_params(encoding, args, n_features=len(feat_idx), model_name=model)
                print(f"  [RUN] dataset={dataset_label} model={model} encoding={encoding}", flush=True)
                try:
                    row = run_one_combination(
                        dataset_label,
                        meta,
                        model,
                        encoding,
                        params,
                        X,
                        y,
                        feat_idx,
                        args,
                        encoding_stats_cache,
                    )
                    rows.append(row)
                    if row.get("status") == "OK":
                        print(
                            f"    OK balanced_acc={float(row['balanced_accuracy']):.4f} "
                            f"acc={float(row['accuracy']):.4f} evals={row.get('evals')}",
                            flush=True,
                        )
                        existing_keys.add(run_key)
                    else:
                        print(f"    {row.get('status')}: {row.get('tag', '')}", flush=True)
                        if row.get("status") == "SKIPPED":
                            existing_keys.add(run_key)
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    row = failure_row(dataset_label, meta, model, encoding, params, X, y, feat_idx, args, exc)
                    rows.append(row)
                    print(f"    FAILED: {exc!r}", flush=True)

                # Write incrementally so long sweeps are not lost if interrupted.
                _write_csv_safely(pd.DataFrame(existing_rows + rows), out_path)

    df = pd.DataFrame(existing_rows + rows)
    _write_csv_safely(df, out_path)

    elapsed = time.perf_counter() - t_all
    ok = int((df.get("status") == "OK").sum()) if "status" in df else 0
    failed = int((df.get("status") == "FAILED").sum()) if "status" in df else 0
    skipped = int((df.get("status") == "SKIPPED").sum()) if "status" in df else 0

    print("\nSummary")
    print(f"  rows    : {len(df)}")
    print(f"  OK      : {ok}")
    print(f"  FAILED  : {failed}")
    print(f"  SKIPPED : {skipped}")
    print(f"  seconds : {elapsed:.2f}")
    print(f"  saved   : {out_path}")

    if ok:
        summary = (
            df[df["status"] == "OK"]
            .groupby(["model", "encoding"], dropna=False)["balanced_accuracy"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        with pd.option_context("display.max_rows", 40, "display.max_colwidth", 60, "display.width", 160):
            print("\nTop mean balanced accuracy by model/encoding:")
            print(summary.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
