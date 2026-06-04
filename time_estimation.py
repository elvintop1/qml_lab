from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Sequence, List, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
except Exception:  # pragma: no cover
    QuantumCircuit = Any  # type: ignore
    transpile = None


_ONE_Q_GATES = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "sx", "sxdg",
    "rx", "ry", "rz",
    "u", "u1", "u2", "u3", "p",
    "id",
}
_TWO_Q_GATES = {
    "cx", "cz", "cy", "swap", "iswap",
    "ecr",
    "rxx", "ryy", "rzz", "rzx",
}


# Default basis gate-set used for *gate-based* accounting.
# This is intentionally hardware-agnostic but close to common IBM-style
# transpiled circuits (single-qubit RZ/SX/X + CX entanglers).
_DEFAULT_BASIS_GATES: Tuple[str, ...] = ("rz", "sx", "x", "cx")


def _canon_basis_gates(basis_gates: Optional[Sequence[str]]) -> List[str]:
    if not basis_gates:
        return list(_DEFAULT_BASIS_GATES)
    return [str(g).strip().lower() for g in basis_gates if str(g).strip()]


def prepare_circuit_for_gate_accounting(
    qc,
    *,
    basis_gates: Optional[Sequence[str]] = None,
    optimization_level: int = 0,
    decompose_reps: int = 10,
) -> tuple[Any, Dict[str, Any]]:
    """Return a circuit suitable for *gate-count* accounting.

    Why this exists
    ---------------
    Many encodings (e.g., amplitude / histogram / sparse amplitude) use
    high-level Qiskit instructions such as ``StatePreparation``. If you call
    ``qc.count_ops()`` directly, Qiskit will count that as a single operation
    (e.g., ``state_preparation``), which *dramatically underestimates* the
    true 1Q/2Q gate cost.

    This helper attempts to:
      1) recursively decompose composite instructions, then
      2) transpile to a 1Q/2Q basis gate set,
    so that counts/depth reflect the internal implementation.

    The returned ``meta`` dict is JSON-serializable and can be stored in
    thesis/report artifacts for traceability.
    """

    meta: Dict[str, Any] = {
        "prepared": False,
        "decompose_reps": int(decompose_reps),
        "optimization_level": int(optimization_level),
        "basis_gates": _canon_basis_gates(basis_gates),
    }

    # If Qiskit isn't available, return the circuit as-is.
    if transpile is None:
        meta["note"] = "Qiskit transpile not available; using raw circuit."
        return qc, meta

    # Defensive copy.
    qct = qc
    try:
        qct = qc.copy()
    except Exception:
        qct = qc

    # 1) Decompose nested instructions (e.g., to_instruction(), inverse(),
    #    StatePreparation, PauliEvolutionGate, etc.).
    try:
        reps = max(0, int(decompose_reps))
        for _ in range(reps):
            try:
                before = len(getattr(qct, "data", []) or [])
            except Exception:
                before = None
            qct2 = qct.decompose()
            qct = qct2
            try:
                after = len(getattr(qct, "data", []) or [])
            except Exception:
                after = None
            if (before is not None) and (after is not None) and (after == before):
                break
    except Exception as e:
        meta["decompose_error"] = str(e)

    # 2) Transpile to basis gates so that multi-qubit operations are unrolled
    #    to 1Q/2Q gates. Keep optimization low for more faithful counting.
    try:
        qct = transpile(
            qct,
            basis_gates=meta["basis_gates"],
            optimization_level=int(optimization_level),
        )
        meta["prepared"] = True
    except Exception as e:
        meta["transpile_error"] = str(e)

    return qct, meta


@dataclass
class GateTimeModel:
    """
    Simple time model to convert circuit structure to an estimated QPU execution time.

    WARNING:
      This is an *estimate*, not a measurement. Real runtime depends on:
        - scheduling/parallelism (depth), not just gate counts
        - backend calibration (gate durations)
        - dynamic circuit features (mid-circuit measurement/reset)
        - batching, parameter binding, classical control latency, etc.

    Use this primarily to "scale down" simulator wall-time to a more realistic order-of-magnitude
    for hardware *when presenting results*, and always report your assumptions.
    """
    t_1q: float = 35e-9
    t_2q: float = 350e-9
    t_meas: float = 2e-6
    t_reset: float = 2e-6
    t_other: float = 35e-9

    # per-shot classical overhead (very rough)
    t_shot_overhead: float = 0.0


def _count_ops(qc) -> Dict[str, int]:
    try:
        c = qc.count_ops()
        return {str(k): int(v) for k, v in c.items()}
    except Exception:
        return {}


def gate_summary(
    qc,
    *,
    transpile_to_basis: bool = True,
    basis_gates: Optional[Sequence[str]] = None,
    optimization_level: int = 0,
    decompose_reps: int = 10,
) -> Dict[str, Any]:
    """Return a structural summary of a circuit suitable for thesis/report tables.

    Key point
    ---------
    Several encodings in this repo (notably amplitude / histogram / sparse amplitude)
    use Qiskit's high-level ``StatePreparation`` instruction. Counting ops on the
    raw circuit would treat ``state_preparation`` as a single op and under-estimate
    the real 1Q/2Q gate cost.

    Therefore, by default this function *prepares* the circuit for gate accounting by:
      - recursively decomposing composite instructions, then
      - transpiling to a 1Q/2Q basis (default: rz/sx/x/cx).

    If you want the raw counts as-built, set ``transpile_to_basis=False``.
    """

    meta = None
    qc_eff = qc
    if transpile_to_basis:
        qc_eff, meta = prepare_circuit_for_gate_accounting(
            qc,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            decompose_reps=decompose_reps,
        )

    ops = _count_ops(qc_eff)

    # Basic counts
    n1 = sum(v for k, v in ops.items() if k in _ONE_Q_GATES)
    n2 = sum(v for k, v in ops.items() if k in _TWO_Q_GATES)
    n_cx = int(ops.get("cx", 0))
    n_meas = int(ops.get("measure", 0))
    n_reset = int(ops.get("reset", 0))

    # Anything not categorized above (excluding measurement/reset/barrier).
    n_other = sum(
        v
        for k, v in ops.items()
        if (k not in _ONE_Q_GATES and k not in _TWO_Q_GATES and k not in ("measure", "reset", "barrier"))
    )

    # Total operation counts.
    op_count = int(sum(v for k, v in ops.items() if k != "barrier"))

    # "Gate count" for reporting: unitary-like operations (exclude measure/reset/barrier).
    gate_count = int(op_count - n_meas - n_reset)

    # Depth: in most Qiskit versions this includes non-barrier ops.
    try:
        depth = int(qc_eff.depth())
    except Exception:
        depth = None

    # If supported by the installed Qiskit version, also compute depth excluding measurement/reset.
    depth_no_meas = None
    try:
        depth_no_meas = int(
            qc_eff.depth(
                filter_function=lambda inst: getattr(getattr(inst, "operation", None), "name", "")
                not in ("measure", "reset", "barrier")
            )
        )
    except Exception:
        depth_no_meas = None

    try:
        n_qubits = int(qc_eff.num_qubits)
    except Exception:
        n_qubits = None

    out = {
        # Width/depth
        "depth": depth,
        "depth_no_meas": depth_no_meas,
        "n_qubits": n_qubits,

        # Aggregates
        "op_count": op_count,
        "gate_count": gate_count,

        # Gate family counts
        "n_1q": int(n1),
        "n_2q": int(n2),
        "n_cx": int(n_cx),
        "n_meas": int(n_meas),
        "n_reset": int(n_reset),
        "n_other": int(n_other),

        # Raw op histogram
        "ops": ops,
    }

    if meta is not None:
        # Keep for traceability in JSON reports.
        out["gate_accounting"] = meta

    return out


def estimate_circuit_time_counts(
    qc,
    model: GateTimeModel,
    *,
    meas_parallel: bool = True,
    reset_parallel: bool = True,
) -> float:
    """Gate-count runtime estimate.

    This is "gate-based" in the sense of counting 1-qubit and 2-qubit gates and
    multiplying by assumed gate durations.

    By default, we treat measurement/reset as *parallel layers* (i.e., we only
    pay one measurement duration if the circuit contains any measurement), since
    real hardware typically reads out multiple qubits in parallel.

    If you want the strictest possible upper bound, set meas_parallel=False and
    reset_parallel=False.
    """
    ops = _count_ops(qc)
    n1 = sum(v for k, v in ops.items() if k in _ONE_Q_GATES)
    n2 = sum(v for k, v in ops.items() if k in _TWO_Q_GATES)

    meas_ops = int(ops.get("measure", 0))
    reset_ops = int(ops.get("reset", 0))

    n_meas = 1 if (meas_parallel and meas_ops > 0) else meas_ops
    n_reset = 1 if (reset_parallel and reset_ops > 0) else reset_ops

    n_other = sum(
        v
        for k, v in ops.items()
        if (k not in _ONE_Q_GATES and k not in _TWO_Q_GATES and k not in ("measure", "reset", "barrier"))
    )
    return (
        float(n1) * float(model.t_1q)
        + float(n2) * float(model.t_2q)
        + float(n_meas) * float(model.t_meas)
        + float(n_reset) * float(model.t_reset)
        + float(n_other) * float(model.t_other)
    )


def estimate_circuit_time_depth(qc, model: GateTimeModel) -> float:
    """
    Depth-based runtime estimate: depth * max(t_1q, t_2q) + t_meas.
    This is often closer to hardware than pure counts (still very rough).
    """
    try:
        depth = int(qc.depth())
    except Exception:
        depth = 0
    layer_time = float(max(model.t_1q, model.t_2q))
    # assume one measurement layer at end if the circuit measures
    ops = _count_ops(qc)
    has_meas = int(ops.get("measure", 0)) > 0
    return depth * layer_time + (model.t_meas if has_meas else 0.0)


def estimate_qpu_time(
    qc,
    *,
    evals: int,
    shots: Optional[int],
    model: Optional[GateTimeModel] = None,
    method: str = "depth",
    assumed_shots_if_none: int = 1024,
    meas_parallel: bool = True,
    prepare_circuit: bool = True,
    basis_gates: Optional[Sequence[str]] = None,
    optimization_level: int = 0,
    decompose_reps: int = 10,
) -> float:
    """
    Estimate total QPU runtime for repeated execution of a representative circuit.

    Parameters
    ----------
    qc:
        Representative circuit (one "evaluation").
    evals:
        Number of circuit evaluations (e.g., kernel entries, objective calls * batch_size).
    shots:
        Shots per evaluation. If None, uses assumed_shots_if_none.
    method:
        'depth' (default) or 'counts'
    """
    model = model or GateTimeModel()
    shots_eff = int(assumed_shots_if_none if shots is None else shots)
    if shots_eff <= 0:
        shots_eff = 1

    qc_eff = qc
    if prepare_circuit:
        qc_eff, _meta = prepare_circuit_for_gate_accounting(
            qc,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            decompose_reps=decompose_reps,
        )

    if method == "counts":
        t_eval = estimate_circuit_time_counts(qc_eff, model, meas_parallel=meas_parallel)
    else:
        t_eval = estimate_circuit_time_depth(qc_eff, model)

    t_shot = t_eval + model.t_shot_overhead
    return float(evals) * float(shots_eff) * float(t_shot)


def scale_factor_from_simulator(sim_seconds: float, est_qpu_seconds: float) -> float:
    """
    Factor to multiply simulator wall-clock time by to obtain a "hardware-estimated" time.

    Example:
      sim_seconds = 120.0
      est_qpu_seconds = 0.8
      scale = 0.0067
      scaled_time = sim_seconds * scale = 0.8
    """
    sim_seconds = float(sim_seconds)
    est_qpu_seconds = float(est_qpu_seconds)
    if sim_seconds <= 0:
        return 1.0
    return est_qpu_seconds / sim_seconds


def scale_timings(timings: Dict[str, float], scale: float) -> Dict[str, float]:
    """Return a new timings dict scaled by a constant factor."""
    s = float(scale)
    return {k: float(v) * s for k, v in (timings or {}).items()}
