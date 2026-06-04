from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
    _HAVE_AER = True
except Exception:
    AerSimulator = None
    _HAVE_AER = False

from ..encodings.helpers import build_encoding
from .protocols import RunResult
from .torch_statevector import (
    apply_qnn_ansatz,
    bce_loss_from_prob,
    encoded_state_matrix,
    make_parameter,
    probability_one,
    to_torch_labels,
    to_torch_states,
)


def spsa(
    objective: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    *,
    steps: int = 200,
    a: float = 0.1,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: int = 0,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
) -> np.ndarray:
    """
    SPSA optimizer (2 objective evaluations per step).
    """
    rng = np.random.default_rng(seed)
    theta = theta0.astype(float).copy()
    A = steps / 10.0
    for k in range(1, steps + 1):
        ak = a / (k + A) ** alpha
        ck = c / (k) ** gamma
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        yp = objective(theta + ck * delta)
        ym = objective(theta - ck * delta)
        ghat = (yp - ym) / (2.0 * ck) * delta
        theta = theta - ak * ghat
        if callback and (k == 1 or k % max(1, steps // 10) == 0):
            callback(k, objective(theta), theta)
    return theta


@dataclass
class QNNConfig:
    encoding_name: str
    encoding_params: Dict
    feat_idx: List[int]

    depth: int = 2
    readout: int = 0
    shots: Optional[int] = None

    epochs: int = 150
    optimizer: str = "backprop"  # backprop | spsa
    lr: float = 0.05
    weight_decay: float = 0.0
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    batch_size: int = 32

    seed: int = 42
    max_train: Optional[int] = None
    max_test: Optional[int] = None


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


def _scale_train_test(Xtr_raw: np.ndarray, Xte_raw: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0.0, 2 * np.pi))
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)
    return Xtr.astype(np.float32), Xte.astype(np.float32), scaler


def _ansatz_layer(qc: QuantumCircuit, thetas: np.ndarray, idx0: int, n: int) -> int:
    idx = idx0
    for q in range(n):
        qc.ry(float(thetas[idx]), q); idx += 1
        qc.rz(float(thetas[idx]), q); idx += 1
    # ring entanglement
    if n >= 2:
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.cx(n - 1, 0)
    return idx


def build_qnn_circuit(enc, x: np.ndarray, thetas: np.ndarray, depth: int) -> QuantumCircuit:
    n = enc.num_qubits
    qc = QuantumCircuit(n, name="QNN")
    qc.compose(enc.build(x), range(n), inplace=True)
    idx = 0
    for _ in range(int(depth)):
        idx = _ansatz_layer(qc, thetas, idx, n)
    return qc


def predict_proba_single(
    enc,
    x: np.ndarray,
    thetas: np.ndarray,
    *,
    depth: int,
    readout: int,
    shots: Optional[int] = None,
) -> float:
    """
    p(class=1) = (1 - <Z_readout>)/2.
    """
    qc = build_qnn_circuit(enc, x, thetas, depth)

    if shots is None:
        psi = Statevector.from_instruction(qc).data
        probs = np.abs(psi) ** 2
        idx = np.arange(probs.size, dtype=np.int64)
        bit = (idx >> int(readout)) & 1
        expZ = float(probs[bit == 0].sum() - probs[bit == 1].sum())
        return 0.5 * (1.0 - expZ)

    if not _HAVE_AER:
        raise RuntimeError("qiskit-aer not available (needed for shots-based execution).")
    be = AerSimulator()
    # Measure only the readout qubit
    meas = QuantumCircuit(qc.num_qubits, 1)
    meas.compose(qc, inplace=True)
    meas.measure(int(readout), 0)
    res = be.run(meas, shots=int(shots)).result()
    counts = res.get_counts()
    p1 = counts.get("1", 0) / float(shots)
    return float(p1)


def _logistic_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _train_qnn_backprop(
    enc,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    theta0: np.ndarray,
    *,
    depth: int,
    readout: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
):
    import torch

    torch.manual_seed(int(seed))
    train_states = to_torch_states(encoded_state_matrix(enc, Xtr))
    test_states = to_torch_states(encoded_state_matrix(enc, Xte))
    y_train = to_torch_labels(ytr)

    theta = make_parameter(theta0)
    opt = torch.optim.Adam([theta], lr=float(lr), weight_decay=float(weight_decay))
    bsz = int(batch_size) if batch_size and int(batch_size) > 0 else len(ytr)
    bsz = max(1, min(bsz, len(ytr)))

    final_loss = float("nan")
    for _epoch in range(int(epochs)):
        perm = torch.randperm(len(ytr))
        for start in range(0, len(ytr), bsz):
            idx = perm[start : start + bsz]
            states = apply_qnn_ansatz(
                train_states.index_select(0, idx),
                theta,
                num_qubits=enc.num_qubits,
                depth=depth,
            )
            p1 = probability_one(states, readout)
            loss = bce_loss_from_prob(p1, y_train.index_select(0, idx))
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        states = apply_qnn_ansatz(test_states, theta, num_qubits=enc.num_qubits, depth=depth)
        p1 = probability_one(states, readout).detach().cpu().numpy().astype(float)

    return theta.detach().cpu().numpy().astype(float), p1, final_loss, bsz


def run_qnn_train_test(
    X_raw: np.ndarray,
    y: np.ndarray,
    cfg: QNNConfig,
    test_size: float = 0.30,
    seed: int = 42,
    *,
    return_result: bool = False,
    capture_circuits: bool = False,
    X_test_raw: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
):
    """
    Binary QNN classifier with SPSA training.

    Practical notes:
      - For simulators, keep qubits <= ~10 and keep train set small.
      - cfg.max_train/cfg.max_test exist to keep experiments feasible.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if X_test_raw is not None:
        if y_test is None:
            raise ValueError("y_test must be provided with X_test_raw.")
        X_test_raw = np.asarray(X_test_raw, dtype=float)
        y_test = np.asarray(y_test, dtype=int).ravel()
        all_labels = np.unique(np.concatenate([y, y_test]))
    else:
        all_labels = np.unique(y)
    if set(all_labels) - {0, 1}:
        raise ValueError("QNN in this phase-2 implementation supports binary y in {0,1} only.")

    Xsel = X_raw[:, cfg.feat_idx]
    if X_test_raw is None:
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            Xsel, y, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        Xtr_raw = Xsel
        Xte_raw = X_test_raw[:, cfg.feat_idx]
        ytr = y
        yte = y_test

    if cfg.max_train is not None:
        Xtr_raw, ytr = _stratified_subsample(Xtr_raw, ytr, int(cfg.max_train), seed=seed)
    if cfg.max_test is not None:
        Xte_raw, yte = _stratified_subsample(Xte_raw, yte, int(cfg.max_test), seed=seed + 1)

    Xtr, Xte, scaler = _scale_train_test(Xtr_raw, Xte_raw)

    enc, num_qubits, _ = build_encoding(
        cfg.encoding_name,
        n_features=Xtr.shape[1],
        params=cfg.encoding_params,
        X_fit=Xtr,
        y_fit=ytr,
    )

    depth = int(cfg.depth)
    num_params = depth * num_qubits * 2
    rng = np.random.default_rng(cfg.seed)
    theta0 = rng.uniform(-0.1, 0.1, size=(num_params,))

    # Training objective
    idx_all = np.arange(len(Xtr))
    bsz = int(cfg.batch_size) if cfg.batch_size and cfg.batch_size > 0 else len(Xtr)
    bsz = min(bsz, len(Xtr))

    timings = {}
    t0 = time.perf_counter()
    optimizer_name = str(getattr(cfg, "optimizer", "backprop") or "backprop").lower().strip()
    use_backprop = optimizer_name in {"backprop", "adam", "autograd"} and cfg.shots is None

    if use_backprop:
        theta_opt, p1, final_loss, bsz = _train_qnn_backprop(
            enc,
            Xtr,
            ytr,
            Xte,
            theta0,
            depth=depth,
            readout=cfg.readout,
            epochs=int(cfg.epochs),
            batch_size=bsz,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            seed=int(cfg.seed),
        )
        timings["train_loss_final"] = float(final_loss)
    else:
        optimizer_name = "spsa"

        def objective(theta: np.ndarray) -> float:
            if bsz < len(Xtr):
                idx = rng.choice(idx_all, size=bsz, replace=False)
            else:
                idx = idx_all
            ps = np.array(
                [predict_proba_single(enc, Xtr[i], theta, depth=depth, readout=cfg.readout, shots=cfg.shots) for i in idx],
                dtype=float,
            )
            return _logistic_loss(ps, ytr[idx])

        theta_opt = spsa(
            objective,
            theta0,
            steps=int(cfg.epochs),
            a=float(cfg.spsa_a),
            c=float(cfg.spsa_c),
            seed=int(cfg.seed),
        )
    timings["train"] = time.perf_counter() - t0

    # Predict on test set
    t0 = time.perf_counter()
    if not use_backprop:
        p1 = np.array(
            [predict_proba_single(enc, x, theta_opt, depth=depth, readout=cfg.readout, shots=cfg.shots) for x in Xte],
            dtype=float,
        )
    yhat = (p1 >= 0.5).astype(int)
    timings["predict"] = time.perf_counter() - t0

    acc = float(accuracy_score(yte, yhat))
    cm = confusion_matrix(yte, yhat)

    tag = (
        f"QNN enc={cfg.encoding_name} depth={depth} qubits={num_qubits} "
        f"epochs={cfg.epochs} optimizer={optimizer_name} lr={cfg.lr} shots={cfg.shots} batch={bsz}"
    )

    if not return_result:
        return acc, cm, tag

    circuits = None
    if capture_circuits:
        try:
            from qiskit import QuantumCircuit
            base_c = build_qnn_circuit(enc, Xtr[0], theta_opt, depth)
            qc = QuantumCircuit(base_c.num_qubits, 1, name="QNN_Infer")
            qc.compose(base_c, inplace=True)
            qc.measure(int(cfg.readout), 0)
            circuits = [qc]
        except Exception:
            circuits = None

    # Backprop uses exact statevector batches; SPSA uses two objective evaluations per step.
    evals_train = int(int(cfg.epochs) * len(Xtr)) if use_backprop else int(2 * int(cfg.epochs) * bsz)
    evals_pred = int(len(Xte))
    evals_total = int(evals_train + evals_pred)

    y_prob = np.vstack([1.0 - p1, p1]).T

    result = RunResult(
        y_true=yte,
        y_pred=yhat,
        y_prob=y_prob,
        K_train=None,
        K_test=None,
        y_train=None,
        circuits=circuits,
        model_name="QNN",
        encoding_name=cfg.encoding_name,
        encoding_params=cfg.encoding_params,
        feat_idx=cfg.feat_idx,
        num_params=num_params,
        shots=cfg.shots,
        evals=evals_total,
        timings=timings,
        extras={
            "tag": tag,
            "confusion_matrix": cm,
            "num_qubits": num_qubits,
            "optimizer": optimizer_name,
            "learning_rate": float(cfg.lr),
            "evals_train": evals_train,
            "evals_pred": evals_pred,
            "batch_size_effective": bsz,
        },
    )
    return acc, cm, tag, result
