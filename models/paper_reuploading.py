from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PaperReuploadingConfig:
    qubits: int = 1
    layers: int = 2
    entanglement: bool = False
    method: str = "L-BFGS-B"
    maxiter: int | None = 120
    maxfun: int | None = None
    ftol: float = 1e-9
    seed: int = 30


@dataclass
class PaperReuploadingResult:
    theta: np.ndarray
    alpha: np.ndarray
    weight: np.ndarray
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    success: bool
    message: str
    nfev: int
    nit: int
    timings: Dict[str, float]


def representative_states(classes: int) -> np.ndarray:
    """One-qubit label representatives used in arXiv:1907.02085."""
    reprs = np.zeros((int(classes), 2), dtype=np.complex128)
    if classes == 2:
        reprs[0] = np.array([1.0, 0.0])
        reprs[1] = np.array([0.0, 1.0])
    elif classes == 3:
        reprs[0] = np.array([1.0, 0.0])
        reprs[1] = np.array([0.5, np.sqrt(3.0) / 2.0])
        reprs[2] = np.array([0.5, -np.sqrt(3.0) / 2.0])
    elif classes == 4:
        reprs[0] = np.array([1.0, 0.0])
        reprs[1] = np.array([1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)])
        reprs[2] = np.array(
            [1.0 / np.sqrt(3.0), np.exp(1j * 2.0 * np.pi / 3.0) * np.sqrt(2.0 / 3.0)]
        )
        reprs[3] = np.array(
            [1.0 / np.sqrt(3.0), np.exp(-1j * 2.0 * np.pi / 3.0) * np.sqrt(2.0 / 3.0)]
        )
    else:
        raise ValueError("The paper runner supports 2, 3, or 4 classes.")
    return reprs


def _validate_config(config: PaperReuploadingConfig, x: np.ndarray) -> None:
    if config.qubits not in {1, 2, 4}:
        raise ValueError("The paper ansatz supports qubits in {1, 2, 4}.")
    if int(config.layers) < 1:
        raise ValueError("layers must be >= 1.")
    if x.shape[1] not in {2, 3, 4}:
        raise ValueError("The paper datasets are 2D, 3D, or 4D.")
    if config.entanglement and config.qubits == 1:
        raise ValueError("entanglement requires 2 or 4 qubits.")


def _init_params(
    rng: np.random.Generator,
    qubits: int,
    layers: int,
    dim: int,
    classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_width = 6 if dim == 4 else 3
    theta = rng.random((qubits, layers, theta_width))
    alpha = rng.random((qubits, layers, dim))
    weight = np.ones((classes, qubits), dtype=float)
    return theta, alpha, weight


def _pack(theta: np.ndarray, alpha: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.concatenate([theta.ravel(), alpha.ravel(), weight.ravel()])


def _unpack(
    params: np.ndarray,
    *,
    qubits: int,
    layers: int,
    dim: int,
    classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_width = 6 if dim == 4 else 3
    theta_size = qubits * layers * theta_width
    alpha_size = qubits * layers * dim
    theta = params[:theta_size].reshape(qubits, layers, theta_width)
    alpha = params[theta_size : theta_size + alpha_size].reshape(qubits, layers, dim)
    weight = params[theta_size + alpha_size :].reshape(classes, qubits)
    return theta, alpha, weight


def _apply_u3_batch(states: np.ndarray, qubit: int, phi: np.ndarray) -> None:
    qubits = int(np.log2(states.shape[1]))
    c = np.cos(phi[:, 0] / 2.0)
    s = np.sin(phi[:, 0] / 2.0)
    e_phi = np.exp(1j * phi[:, 1] / 2.0)
    e_phi_s = np.conj(e_phi)
    e_lambda = np.exp(1j * phi[:, 2] / 2.0)
    e_lambda_s = np.conj(e_lambda)

    for k in range(2 ** (qubits - 1)):
        s0 = k % (2**qubit) + 2 * (k - k % (2**qubit))
        s1 = s0 + 2**qubit
        a0 = states[:, s0].copy()
        a1 = states[:, s1].copy()
        states[:, s0] = c * e_phi * e_lambda * a0 - s * e_phi * e_lambda_s * a1
        states[:, s1] = s * e_phi_s * e_lambda * a0 + c * e_phi_s * e_lambda_s * a1


def _apply_cz_batch(states: np.ndarray, q0: int, q1: int) -> None:
    if q0 == q1:
        raise ValueError("CZ qubits must be different.")
    mask = [idx for idx in range(states.shape[1]) if ((idx >> q0) & 1) and ((idx >> q1) & 1)]
    states[:, mask] *= -1.0


def _phi_for_low_dim(theta_row: np.ndarray, alpha_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi = np.broadcast_to(theta_row[:3], (x.shape[0], 3)).astype(float, copy=True)
    dim = x.shape[1]
    phi[:, :dim] += x * alpha_row[:dim]
    return phi


def _phi_pair_for_4d(theta_row: np.ndarray, alpha_row: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    phi_a = np.broadcast_to(theta_row[:3], (x.shape[0], 3)).astype(float, copy=True)
    phi_b = np.broadcast_to(theta_row[3:6], (x.shape[0], 3)).astype(float, copy=True)
    phi_a[:, 0] += alpha_row[0] * x[:, 0]
    phi_a[:, 1] += alpha_row[1] * x[:, 1]
    phi_b[:, 0] += alpha_row[2] * x[:, 2]
    phi_b[:, 1] += alpha_row[3] * x[:, 3]
    return phi_a, phi_b


def _apply_entangling_layer(states: np.ndarray, qubits: int, layer: int) -> None:
    if qubits == 2:
        _apply_cz_batch(states, 0, 1)
    elif qubits == 4 and layer % 2 == 0:
        _apply_cz_batch(states, 0, 1)
        _apply_cz_batch(states, 2, 3)
    elif qubits == 4:
        _apply_cz_batch(states, 1, 2)
        _apply_cz_batch(states, 0, 3)
    else:
        raise ValueError("Entanglement is defined only for 2 or 4 qubits.")


def _simulate_states(theta: np.ndarray, alpha: np.ndarray, x: np.ndarray, entanglement: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    qubits, layers, theta_width = theta.shape
    states = np.zeros((x.shape[0], 2**qubits), dtype=np.complex128)
    states[:, 0] = 1.0

    for layer in range(layers):
        for qubit in range(qubits):
            if theta_width == 3:
                _apply_u3_batch(states, qubit, _phi_for_low_dim(theta[qubit, layer], alpha[qubit, layer], x))
            else:
                phi_a, phi_b = _phi_pair_for_4d(theta[qubit, layer], alpha[qubit, layer], x)
                _apply_u3_batch(states, qubit, phi_a)
                _apply_u3_batch(states, qubit, phi_b)
        if entanglement and layer < layers - 1:
            _apply_entangling_layer(states, qubits, layer)
    return states


def _label_fidelity_for_qubit(states: np.ndarray, qubit: int, label_state: np.ndarray) -> np.ndarray:
    qubits = int(np.log2(states.shape[1]))
    out = np.zeros(states.shape[0], dtype=float)
    bra0, bra1 = np.conj(label_state[0]), np.conj(label_state[1])
    for k in range(2 ** (qubits - 1)):
        s0 = k % (2**qubit) + 2 * (k - k % (2**qubit))
        s1 = s0 + 2**qubit
        projected = bra0 * states[:, s0] + bra1 * states[:, s1]
        out += np.abs(projected) ** 2
    return out


def fidelities(
    theta: np.ndarray,
    alpha: np.ndarray,
    x: np.ndarray,
    reprs: np.ndarray,
    *,
    entanglement: bool,
) -> np.ndarray:
    states = _simulate_states(theta, alpha, x, entanglement)
    classes = reprs.shape[0]
    qubits = theta.shape[0]
    out = np.empty((x.shape[0], classes, qubits), dtype=float)
    for cls in range(classes):
        for qubit in range(qubits):
            out[:, cls, qubit] = _label_fidelity_for_qubit(states, qubit, reprs[cls])
    return out


def weighted_scores(
    theta: np.ndarray,
    alpha: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    reprs: np.ndarray,
    *,
    entanglement: bool,
) -> np.ndarray:
    fids = fidelities(theta, alpha, x, reprs, entanglement=entanglement)
    return np.einsum("ncq,cq->nc", fids, weight)


def _target_matrix(y: np.ndarray, classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if classes == 2:
        target = np.zeros((len(y), classes), dtype=float)
    elif classes == 3:
        target = np.full((len(y), classes), 0.25, dtype=float)
    elif classes == 4:
        target = np.full((len(y), classes), 1.0 / 3.0, dtype=float)
    else:
        raise ValueError("The paper runner supports 2, 3, or 4 classes.")
    target[np.arange(len(y)), y] = 1.0
    return target


def weighted_fidelity_loss(
    theta: np.ndarray,
    alpha: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    reprs: np.ndarray,
    *,
    entanglement: bool,
) -> float:
    scores = weighted_scores(theta, alpha, weight, x, reprs, entanglement=entanglement)
    target = _target_matrix(y, reprs.shape[0])
    return float(0.5 * np.mean(np.sum((scores - target) ** 2, axis=1)))


def predict(
    theta: np.ndarray,
    alpha: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    reprs: np.ndarray,
    *,
    entanglement: bool,
) -> np.ndarray:
    scores = weighted_scores(theta, alpha, weight, x, reprs, entanglement=entanglement)
    return np.argmax(scores, axis=1).astype(int)


def fit_paper_reuploading_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: PaperReuploadingConfig,
) -> PaperReuploadingResult:
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("SciPy is required for the paper L-BFGS-B runner. Install scipy in the venv.") from exc

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)
    _validate_config(config, x_train)

    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError("Train and test dimensions do not match.")
    classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    reprs = representative_states(classes)
    rng = np.random.default_rng(int(config.seed))
    theta0, alpha0, weight0 = _init_params(rng, config.qubits, config.layers, x_train.shape[1], classes)
    params0 = _pack(theta0, alpha0, weight0)

    objective_calls = 0

    def objective(params: np.ndarray) -> float:
        nonlocal objective_calls
        objective_calls += 1
        theta, alpha, weight = _unpack(
            params,
            qubits=config.qubits,
            layers=config.layers,
            dim=x_train.shape[1],
            classes=classes,
        )
        return weighted_fidelity_loss(
            theta,
            alpha,
            weight,
            x_train,
            y_train,
            reprs,
            entanglement=config.entanglement,
        )

    options: Dict[str, float | int] = {"ftol": float(config.ftol)}
    if config.maxiter is not None:
        options["maxiter"] = int(config.maxiter)
    if config.maxfun is not None:
        options["maxfun"] = int(config.maxfun)

    t0 = perf_counter()
    optimized = minimize(objective, params0, method=config.method, options=options)
    train_seconds = perf_counter() - t0
    theta, alpha, weight = _unpack(
        optimized.x,
        qubits=config.qubits,
        layers=config.layers,
        dim=x_train.shape[1],
        classes=classes,
    )

    t1 = perf_counter()
    train_pred = predict(theta, alpha, weight, x_train, reprs, entanglement=config.entanglement)
    test_pred = predict(theta, alpha, weight, x_test, reprs, entanglement=config.entanglement)
    predict_seconds = perf_counter() - t1

    train_loss = weighted_fidelity_loss(
        theta, alpha, weight, x_train, y_train, reprs, entanglement=config.entanglement
    )
    test_loss = weighted_fidelity_loss(theta, alpha, weight, x_test, y_test, reprs, entanglement=config.entanglement)

    return PaperReuploadingResult(
        theta=theta,
        alpha=alpha,
        weight=weight,
        train_loss=train_loss,
        test_loss=test_loss,
        train_accuracy=float(np.mean(train_pred == y_train)),
        test_accuracy=float(np.mean(test_pred == y_test)),
        train_predictions=train_pred,
        test_predictions=test_pred,
        success=bool(optimized.success),
        message=str(optimized.message),
        nfev=int(getattr(optimized, "nfev", objective_calls)),
        nit=int(getattr(optimized, "nit", 0)),
        timings={"train": train_seconds, "predict": predict_seconds},
    )
