from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

try:
    from qiskit import transpile
except Exception:
    transpile = None



def task_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return out



def circuit_stats(
    circuits: List[Any],
    basis_gates: Optional[List[str]] = None,
    coupling_map=None,
    optimization_level: int = 3,
) -> Dict[str, Any]:
    """
    Transpile representative circuits and aggregate stats.
    circuits: list of QuantumCircuit (could be length 1)
    """
    if transpile is None or not circuits:
        return {"note": "No circuits or Qiskit transpile not available."}

    widths, depths, sizes, twoqs, counts_list = [], [], [], [], []
    twoq_set = {"cx", "cz", "rzz", "iswap", "ecr"}
    for qc in circuits:
        tqc = transpile(qc, basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=optimization_level)
        counts = tqc.count_ops()
        counts_list.append({k: int(v) for k, v in counts.items()})
        widths.append(tqc.num_qubits)
        depths.append(tqc.depth())
        sizes.append(tqc.size())
        twoqs.append(sum(int(counts.get(g, 0)) for g in twoq_set))

    def _agg(L):  
        return {"mean": float(np.mean(L)), "min": int(np.min(L)), "max": int(np.max(L))}

    return {
        "num_samples": len(circuits),
        "width_qubits": _agg(widths),
        "depth": _agg(depths),
        "size_ops": _agg(sizes),
        "two_qubit_gates": _agg(twoqs),
        "op_counts_each": counts_list,  
    }



def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def kernel_psd_stats(K: np.ndarray) -> Dict[str, Any]:
    Ksym = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ksym)
    return {
        "min_eig": float(w.min()),
        "num_neg_eigs": int(np.sum(w < -1e-12)),
        "trace": float(np.sum(w)),
        "fro_norm": float(np.linalg.norm(Ksym, ord="fro")),
    }

def kernel_condition(K: np.ndarray, lam: float = 1e-8) -> float:
    Kc = _center(K)
    w = np.linalg.eigvalsh(Kc + lam * np.eye(Kc.shape[0]))
    w = np.clip(w, 1e-12, None)
    return float(w.max() / w.min())

def kernel_effective_rank(K: np.ndarray, eps: float = 1e-6) -> int:
    w = np.linalg.eigvalsh(_center(K))
    thr = max(1e-12, eps * float(np.max(w)))
    return int(np.sum(w > thr))

def kernel_effective_dimension(K: np.ndarray, lam: float = 1e-2) -> float:
    w = np.linalg.eigvalsh(_center(K))
    return float(np.sum(w / (w + lam)))

def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Multi-class kernel-target alignment using one-hot label kernel.
    """
    y = y.reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Y = enc.fit_transform(y)
    L = Y @ Y.T
    Kc = _center(K)
    Lc = _center(L)
    num = float(np.sum(Kc * Lc))
    den = float(np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)) + 1e-12)
    return num / den



def time_block(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return {"seconds": time.perf_counter() - t0, "result": out}

def qsvm_shot_cost(n_tr: int, n_te: int, shots: Optional[int]) -> Dict[str, Any]:
    if not shots:
        return {"kernel_evals_train": n_tr * (n_tr + 1) // 2,
                "kernel_evals_test": n_te * n_tr,
                "kernel_evals_total": n_tr * (n_tr + 1) // 2 + n_te * n_tr,
                "total_shots": 0}
    evals_tr = n_tr * (n_tr + 1) // 2
    evals_te = n_te * n_tr
    total = evals_tr + evals_te
    return {"kernel_evals_train": int(evals_tr),
            "kernel_evals_test": int(evals_te),
            "kernel_evals_total": int(total),
            "total_shots": int(total * shots)}

def variational_eval_cost(n_samples: int, epochs: int, batch_size: int = 0) -> int:
    """
    Approx objective calls for SPSA: ~ 2 * (#batches per epoch) * epochs.
    (#batches) = ceil(n_samples / (batch_size or n_samples))
    """
    if n_samples <= 0 or epochs <= 0:
        return 0
    bsz = batch_size if batch_size and batch_size > 0 else n_samples
    n_batches = (n_samples + bsz - 1) // bsz
    return int(2 * n_batches * epochs)



def summarize_result(
    result,                              
    *,
    circuit_basis_gates: Optional[List[str]] = None,
    circuit_coupling_map=None,
    cond_lambda: float = 1e-8,
    rank_eps: float = 1e-6,
    effdim_lambda: float = 1e-2,
) -> Dict[str, Any]:
    """
    Build a single report dict from a RunResult.
    Skips sections automatically if required artifacts are missing.
    """
    report: Dict[str, Any] = {
        "context": {
            "model_name": result.model_name,
            "encoding_name": result.encoding_name,
            "encoding_params": result.encoding_params,
            "feat_idx": result.feat_idx,
            "num_params": result.num_params,
            "shots": result.shots,
            "evals": result.evals,
        },
        "timings": result.timings or {},
        "task": {},
        "kernel": {},
        "circuit": {},
        "extras": result.extras or {},
    }

    if result.y_true is not None and result.y_pred is not None:
        report["task"] = task_metrics(result.y_true, result.y_pred, result.y_prob)

    if result.K_train is not None:
        K = result.K_train
        report["kernel"]["psd"] = kernel_psd_stats(K)
        report["kernel"]["condition"] = kernel_condition(K, lam=cond_lambda)
        report["kernel"]["effective_rank"] = kernel_effective_rank(K, eps=rank_eps)
        report["kernel"]["effective_dimension"] = kernel_effective_dimension(K, lam=effdim_lambda)
        if result.y_train is not None:
            report["kernel"]["alignment"] = kernel_alignment(K, result.y_train)

        if result.K_test is not None and result.shots is not None and result.y_true is not None:
            n_tr = K.shape[0]
            n_te = result.y_true.shape[0]
            report["kernel"]["shot_cost"] = qsvm_shot_cost(n_tr, n_te, result.shots)

    if result.circuits:
        report["circuit"] = circuit_stats(result.circuits, circuit_basis_gates, circuit_coupling_map)

    return report


def pretty_print_report(report: Dict[str, Any]) -> None:
    """Lightweight printer so you don’t need to open JSON."""
    def p(title, obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, dict):
            print(f"{pad}{title}:")
            for k, v in obj.items():
                p(k, v, indent+1)
        else:
            print(f"{pad}{title}: {obj}")

    for section in ["context", "timings", "task", "kernel", "circuit", "extras"]:
        if section in report and report[section]:
            p(section, report[section], 0)
