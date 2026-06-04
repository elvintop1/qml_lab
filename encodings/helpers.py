from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np

from .registry import get_encoding_cls
try:
    from .hardware_aware import is_hardware_aware_name, resolve_data_dependent_params
except ModuleNotFoundError as exc:
    if "hardware_aware" not in str(exc):
        raise

    def is_hardware_aware_name(name: str) -> bool:
        return False

    def resolve_data_dependent_params(encoding_name: str, params: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        return params

try:
    from .ha_sage import is_ha_sage_name, resolve_ha_sage_params
except ModuleNotFoundError as exc:
    if "ha_sage" not in str(exc):
        raise

    def is_ha_sage_name(name: str) -> bool:
        return False

    def resolve_ha_sage_params(encoding_name: str, params: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        return params

try:
    from .ha_sage_cmtsd import is_ha_sage_cmtsd_name, resolve_ha_sage_cmtsd_params
except ModuleNotFoundError as exc:
    if "ha_sage_cmtsd" not in str(exc):
        raise

    def is_ha_sage_cmtsd_name(name: str) -> bool:
        return False

    def resolve_ha_sage_cmtsd_params(encoding_name: str, params: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        return params


def features_per_qubit(encoding_name: str, params: Optional[Dict[str, Any]] = None) -> int:
    """
    Determine how many classical features map to one qubit.

    Convention in this repo:
      - denseangle: 2 features / qubit (theta, phi)
      - others: 1 feature / qubit
    You can override by passing encoding_params={"features_per_qubit": k}.
    """
    params = params or {}
    if "features_per_qubit" in params:
        return int(params["features_per_qubit"])

    key = str(encoding_name).lower().strip()
    key = key.replace("-", "_").replace(" ", "").replace("_", "")
    if key == "denseangle":
        return 2
    return 1


def infer_num_qubits(encoding_name: str, n_features: int, params: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """
    Infer (num_qubits, features_per_qubit) from encoding choice and feature dimension.
    """
    if n_features <= 0:
        raise ValueError("n_features must be > 0")

    key = str(encoding_name).lower().strip()
    key = key.replace("-", "_").replace(" ", "").replace("_", "")

    # --- Explicit override (used by data re-uploading, integer, onehot, etc.) ---
    # NOTE: build_encoding(...) pops these keys before calling this helper,
    # but we keep the logic here for defensive usage in other contexts.
    for k in ("num_qubits", "n_qubits", "qubits"):
        if params and k in params:
            try:
                nq = int(params[k])
                if nq > 0:
                    return nq, features_per_qubit(key, params)
            except Exception:
                pass

    # --- Amplitude-like encodings ---
    if key in {"amplitude", "histogram", "probability", "probabilityhistogram", "sparse", "sparseamplitude"}:
        # amplitude needs 2**n amplitudes; allow padding/truncation in the encoding
        n_qubits = int(np.ceil(np.log2(max(2, n_features))))
        return n_qubits, n_features  # fpp is not meaningful for amplitude

    # --- Data re-uploading (typically uses a small fixed number of qubits) ---
    if key in {"reupload", "reuploading", "datareuploading"}:
        # default to 1 qubit if not specified
        nq = 1
        if params:
            nq = int(params.get("num_qubits", params.get("n_qubits", nq)))
        nq = max(1, nq)
        return nq, n_features

    # --- Basis: integer and one-hot can choose qubit count via parameters ---
    if key in {"integer", "binaryinteger"}:
        nq = int(params.get("n_bits", params.get("bits", n_features))) if params else int(n_features)
        nq = max(1, nq)
        return nq, 1

    if key in {"onehot"}:
        nq = int(params.get("n_categories", params.get("categories", n_features))) if params else int(n_features)
        nq = max(1, nq)
        return nq, 1

    if is_hardware_aware_name(key):
        target = 8
        if params:
            target = int(params.get("target_qubits", params.get("feature_qubits", target)))
        nq = max(1, min(int(n_features), int(target)))
        return nq, 1

    if is_ha_sage_name(key):
        target = 8
        if params:
            target = int(params.get("target_qubits", params.get("feature_qubits", params.get("latent_qubits", target))))
        nq = max(1, min(int(n_features), int(target)))
        return nq, 1

    if is_ha_sage_cmtsd_name(key):
        target = 8
        if params:
            target = int(params.get("target_qubits", params.get("feature_qubits", params.get("latent_qubits", target))))
        nq = max(1, min(int(n_features), int(target)))
        return nq, 1

    fpp = features_per_qubit(key, params)

    # DenseAngle can safely pad a missing phi angle with zero.  The old helper
    # incorrectly rejected odd feature counts such as wine_13, even though
    # DenseAngleEncoding.build already supported partial theta/phi pairs.
    if key == "denseangle":
        return int(np.ceil(float(n_features) / float(max(1, fpp)))), fpp

    if n_features % fpp != 0:
        raise ValueError(
            f"[encoding={encoding_name}] Need a multiple of features_per_qubit={fpp} features; got n_features={n_features}."
        )
    return n_features // fpp, fpp


def build_encoding(
    encoding_name: str,
    n_features: int,
    params: Optional[Dict[str, Any]] = None,
    *,
    X_fit: Optional[np.ndarray] = None,
    y_fit: Optional[np.ndarray] = None,
):
    """
    Build (encoding_instance, num_qubits, features_per_qubit).
    """
    params = dict(params or {})

    # Keep features_per_qubit in params.  Most encodings accept **kwargs, and
    # HA-SAGE-CMTSD uses it during the C-MTSD projection fit.

    # Allow explicit num_qubits override (needed for data re-uploading, onehot, integer, etc.)
    nq_override = None
    for k in ("num_qubits", "n_qubits", "qubits"):
        if k in params:
            try:
                nq_override = int(params.pop(k))
            except Exception:
                nq_override = None
            break

    if nq_override is not None and nq_override > 0:
        num_qubits = int(nq_override)
        fpp = features_per_qubit(encoding_name, params)
    else:
        num_qubits, fpp = infer_num_qubits(encoding_name, n_features, params)

    params = resolve_data_dependent_params(
        encoding_name,
        params,
        X_fit=X_fit,
        y_fit=y_fit,
        num_qubits=num_qubits,
    )
    params = resolve_ha_sage_params(
        encoding_name,
        params,
        X_fit=X_fit,
        y_fit=y_fit,
        num_qubits=num_qubits,
    )
    params = resolve_ha_sage_cmtsd_params(
        encoding_name,
        params,
        X_fit=X_fit,
        y_fit=y_fit,
        num_qubits=num_qubits,
    )

    Enc = get_encoding_cls(encoding_name)
    enc = Enc(num_qubits=num_qubits, **params)
    return enc, int(num_qubits), int(fpp)
