from __future__ import annotations

"""UCI Single Elder Home Monitoring: Gas and Position (id=799).

Why this loader exists
----------------------
This dataset is large (~444k rows) and therefore useful for demonstrating
*classical* overheads such as:
  - data cleaning / type conversion (timestamp parsing)
  - classical preprocessing (standardization + PCA)
  - *classical* circuit construction / parameter binding for quantum encodings

The dataset page indicates two separate raw files for gas sensors and
position (movement) sensors. The `ucimlrepo` package exposes a unified
`features` dataframe and a `targets` dataframe.

This loader intentionally:
  - is robust to different target shapes (1 target column vs multiple binary columns)
  - supports a default *binary* target suitable for QNN/QCNN implementations
"""

from typing import Optional, Tuple, List, Literal

import numpy as np
from pandas.api.types import is_numeric_dtype


TargetMode = Literal["any", "room", "first"]
TimestampMode = Literal["drop", "numeric", "cyclic"]


def _fallback_y_from_timestamp(ts_col) -> np.ndarray:
    """Derive a *binary* label from timestamps when `ucimlrepo` provides no targets.

    Why this exists
    ---------------
    For UCI id=799, `ucimlrepo` may return `data.targets=None` because the dataset
    is distributed as multiple CSV files (gas measurements and movement/position
    measurements). In that case, supervised labels are not directly available via
    `fetch_ucirepo`.

    We still want a large dataset for encoding-time benchmarks. As a robust
    fallback, we create a proxy target using the "reference" period described on
    the UCI dataset page:

      - reference (no occupancy) measurements: 2020-01-25 .. 2020-02-13

    Label convention
    ---------------
    y=0 -> reference / no-occupancy (timestamp >= 2020-01-25)
    y=1 -> occupancy period (timestamp < 2020-01-25)

    Notes
    -----
    This is intentionally simple and deterministic. If you later join the
    `database_pos.csv` file, you can replace this target with actual movement
    sensors.
    """

    import pandas as pd

    ts_dt = pd.to_datetime(ts_col, errors="coerce", utc=True)
    if ts_dt.isna().mean() > 0.20:
        # In case parsing fails unexpectedly, fall back to numeric conversion.
        ts_seconds = _safe_to_numeric(ts_col).astype(np.float64)
        ref_start = pd.Timestamp("2020-01-25", tz="UTC").timestamp()
        y = (ts_seconds < ref_start).astype(np.int64)
        return y

    ref_start_dt = pd.Timestamp("2020-01-25", tz="UTC")
    y = (ts_dt < ref_start_dt).astype(np.int64).to_numpy()
    return np.asarray(y, dtype=np.int64)


def _safe_to_numeric(col) -> np.ndarray:
    """Convert a pandas Series to float32 as robustly as possible."""
    import pandas as pd

    if is_numeric_dtype(col):
        # Numeric columns can contain missing values (NaN) or inf; fill them
        # deterministically so downstream transformers (StandardScaler/PCA) do not fail.
        v = col.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(v).all():
            finite = np.isfinite(v)
            if finite.any():
                fill = float(np.nanmean(v[finite]))
                if not np.isfinite(fill):
                    fill = 0.0
            else:
                fill = 0.0
            v = np.where(finite, v, fill)
        return v.astype(np.float32, copy=False)

    # Try datetime
    try:
        dt = pd.to_datetime(col, errors="coerce", utc=True)
        ok = dt.notna().to_numpy()
        if ok.mean() > 0.95:  # treat as timestamp
            # seconds since epoch
            v = (dt.view("int64") / 1e9).astype(np.float64)
            # fill missing with median
            med = np.nanmedian(v)
            v = np.where(np.isfinite(v), v, med)
            return v.astype(np.float32)
    except Exception:
        pass

    # Fallback: factorize categorical strings
    try:
        codes, _ = pd.factorize(col.astype(str), sort=True)
        return codes.astype(np.float32)
    except Exception:
        return np.zeros(len(col), dtype=np.float32)


def _timestamp_features_numeric(ts_seconds: np.ndarray) -> np.ndarray:
    """Return a single numeric timestamp feature (float32)."""
    ts = np.asarray(ts_seconds, dtype=np.float64)
    # Use relative seconds to keep magnitude reasonable.
    ts = ts - np.nanmin(ts)
    ts = np.where(np.isfinite(ts), ts, 0.0)
    return ts.astype(np.float32)[:, None]


def _timestamp_features_cyclic(ts_seconds: np.ndarray) -> np.ndarray:
    """Return cyclic features for time-of-day and day-of-week."""
    ts = np.asarray(ts_seconds, dtype=np.float64)
    ts = np.where(np.isfinite(ts), ts, np.nanmedian(ts))

    # seconds in day/week
    day = 86400.0
    week = 7.0 * day
    tod = np.mod(ts, day) / day  # [0,1)
    dow = np.mod(ts, week) / week

    twopi = 2.0 * np.pi
    feats = np.stack([
        np.sin(twopi * tod), np.cos(twopi * tod),
        np.sin(twopi * dow), np.cos(twopi * dow),
    ], axis=1)
    return feats.astype(np.float32)


def _targets_to_y(y_df, mode: TargetMode) -> np.ndarray:
    """Convert a target dataframe to a 1D int64 label vector."""
    import pandas as pd

    if y_df is None:
        raise ValueError(
            "UCI Elder dataset: targets dataframe is None. "
            "Call load_uci_elder_gas_pos(..., target_mode=...) which will "
            "apply a timestamp-based fallback label when targets are missing."
        )
    if hasattr(y_df, "to_numpy"):
        y_arr = y_df.to_numpy()
    else:
        y_arr = np.asarray(y_df)

    # If a single target column exists, try to use it as-is.
    if y_arr.ndim == 1:
        y1 = y_arr
    elif y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y1 = y_arr[:, 0]
    else:
        y1 = None

    if mode == "first":
        if y1 is None:
            y1 = y_arr[:, 0]
        y = pd.Series(y1).astype("int64", errors="ignore")
        return np.asarray(y, dtype=np.int64)

    if mode == "any":
        if y1 is not None:
            # If already binary-ish, binarize.
            y = pd.Series(y1)
            # Strings like "0"/"1" or booleans are handled.
            try:
                y = y.astype(int)
            except Exception:
                y = (y.astype(str).str.lower().isin(["1", "true", "yes"])).astype(int)
            y = (y.to_numpy() != 0).astype(np.int64)
            return y

        # Multi-column: treat as binary movement/occupied flag.
        y_bin = (np.sum(np.asarray(y_arr, dtype=float), axis=1) > 0.0).astype(np.int64)
        return y_bin

    # mode == "room" (multi-class)
    if y1 is not None:
        # If target already looks categorical/integer, use it.
        y = pd.Series(y1)
        try:
            return y.astype(int).to_numpy(dtype=np.int64)
        except Exception:
            codes, _ = pd.factorize(y.astype(str), sort=True)
            return codes.astype(np.int64)

    # Multi-column binary sensors: argmax room index.
    y_float = np.asarray(y_arr, dtype=float)
    idx = np.argmax(y_float, axis=1).astype(np.int64)
    # Rows with all zeros -> class 0 by convention.
    all_zero = (np.sum(y_float, axis=1) <= 0.0)
    idx = np.where(all_zero, 0, idx + 1)  # shift active rooms to 1..K
    return idx


def load_uci_elder_gas_pos(
    *,
    uci_id: int = 799,
    limit: Optional[int] = None,
    seed: int = 0,
    target_mode: TargetMode = "any",
    timestamp_mode: TimestampMode = "numeric",
    standardize: bool = True,
    pca_dim: Optional[int] = None,
    balance: bool = False,
    n_pos: Optional[int] = None,
    n_neg: Optional[int] = None,
    return_feature_names: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the UCI dataset (id=799) into a numeric matrix suitable for QML.

    Parameters
    ----------
    uci_id:
        UCI dataset id (default 799).
    limit:
        Optional cap on number of samples after any balancing.
        If None, uses all available samples.
    target_mode:
        - "any": binary label = 1 if any movement sensor is active at that timestamp.
        - "room": multi-class label derived from argmax of room sensors.
        - "first": take the first target column as label (mostly for debugging).
    timestamp_mode:
        - "drop": drop timestamp entirely.
        - "numeric": convert timestamp -> numeric seconds (relative).
        - "cyclic": convert timestamp -> sin/cos time-of-day and day-of-week.
    standardize:
        Apply StandardScaler before PCA.
    pca_dim:
        Optional PCA dimension. Use small values (e.g., 8/10/12) for quantum circuits.
    balance:
        If True, performs simple undersampling to match positive/negative counts.
        This is only recommended for *model* training subsets; for full-data
        encoding-time benchmarks, keep balance=False.

    Returns
    -------
    X: np.ndarray[float32] shape (n, d)
    y: np.ndarray[int64] shape (n,)
    feature_names: optional list[str]
    """

    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise ImportError(
            "This loader requires 'ucimlrepo'. Install with: pip install -U ucimlrepo"
        ) from e

    import pandas as pd

    ds = fetch_ucirepo(id=int(uci_id))
    X_df = ds.data.features.copy()
    y_df = ds.data.targets.copy() if ds.data.targets is not None else None

    # ---------
    # Targets
    # ---------
    if y_df is None:
        if "timestamp" not in X_df.columns:
            raise ValueError(
                "UCI Elder dataset: ucimlrepo returned targets=None and the features dataframe "
                "does not contain a 'timestamp' column for the fallback target."
            )

        # Fallback: binary label from timestamp.
        # This makes the pipeline runnable without requiring multi-file joins.
        import warnings

        warnings.warn(
            "UCI id=799: ucimlrepo returned data.targets=None. "
            "Using timestamp-based fallback label (occupied vs reference period).",
            RuntimeWarning,
        )
        y = _fallback_y_from_timestamp(X_df["timestamp"])
    else:
        y = _targets_to_y(y_df, mode=str(target_mode))
    y = np.asarray(y, dtype=np.int64).ravel()

    # ---------
    # Features
    # ---------
    feat_cols = list(X_df.columns)
    feats: List[np.ndarray] = []
    feat_names: List[str] = []

    if "timestamp" in X_df.columns:
        ts = _safe_to_numeric(X_df["timestamp"])  # epoch seconds if possible, else factorized
        if timestamp_mode == "drop":
            pass
        elif timestamp_mode == "cyclic":
            feats.append(_timestamp_features_cyclic(ts))
            feat_names.extend(["ts_sin_day", "ts_cos_day", "ts_sin_week", "ts_cos_week"])
        else:  # numeric
            feats.append(_timestamp_features_numeric(ts))
            feat_names.append("timestamp")

        # remove from remaining
        feat_cols = [c for c in feat_cols if c != "timestamp"]

    # all other columns -> numeric
    for c in feat_cols:
        v = _safe_to_numeric(X_df[c])
        feats.append(v[:, None])
        feat_names.append(str(c))

    X = np.concatenate(feats, axis=1).astype(np.float32, copy=False)

    # ---------
    # Optional subsampling / balancing
    # ---------
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(X))

    if balance:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if n_pos is None and n_neg is None:
            # balance to min class count
            m = int(min(len(pos), len(neg)))
            n_pos = m
            n_neg = m
        elif n_pos is None:
            n_pos = int(min(len(pos), max(1, int(n_neg))))
        elif n_neg is None:
            n_neg = int(min(len(neg), max(1, int(n_pos))))

        pos_sel = rng.choice(pos, size=min(len(pos), int(n_pos)), replace=False) if len(pos) else np.array([], dtype=int)
        neg_sel = rng.choice(neg, size=min(len(neg), int(n_neg)), replace=False) if len(neg) else np.array([], dtype=int)
        idx = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(idx)
    elif limit is not None:
        limit = int(limit)
        if limit > 0 and limit < len(X):
            # stratified-ish: sample proportional to class frequency
            classes, counts = np.unique(y, return_counts=True)
            props = counts / counts.sum()
            take = np.maximum(1, np.floor(props * limit).astype(int))
            while take.sum() < limit:
                take[np.argmax(props)] += 1
            while take.sum() > limit:
                i = np.argmax(take)
                if take[i] > 1:
                    take[i] -= 1
                else:
                    break
            sel = []
            for c, k in zip(classes, take):
                pool = np.where(y == c)[0]
                sel.append(rng.choice(pool, size=min(int(k), len(pool)), replace=False))
            idx = np.concatenate(sel)
            rng.shuffle(idx)

    X = X[idx]
    y = y[idx]

    # ---------
    # Standardize + PCA
    # ---------
    if standardize or (pca_dim is not None):
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X).astype(np.float32)

    if pca_dim is not None:
        from sklearn.decomposition import PCA

        k = int(pca_dim)
        if k <= 0:
            raise ValueError("pca_dim must be > 0")
        X = PCA(n_components=k, random_state=int(seed)).fit_transform(X).astype(np.float32)
        feat_names = [f"pca_{i}" for i in range(X.shape[1])]

    if return_feature_names:
        return X, y, feat_names
    return X, y
