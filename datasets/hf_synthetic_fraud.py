from __future__ import annotations

from typing import Optional, Tuple, List, Union

import numpy as np


TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_hf_synthetic_fraud(
    *,
    dataset_id: str = "purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection",
    split: str = "train",
    n_samples: int = 2000,
    balance: bool = True,
    n_fraud: Optional[int] = None,
    n_legit: Optional[int] = None,
    seed: int = 0,
    include_isFlaggedFraud: bool = False,
    include_names: bool = False,
    type_encoding: str = "onehot",   # "onehot" | "ordinal"
    standardize: bool = True,
    pca_dim: Optional[int] = None,
    num_proc: Optional[int] = None,
    return_feature_names: bool = False,
):
    """
    Load the Hugging Face dataset 'Synthetic-Financial-Datasets-For-Fraud-Detection' and build a numeric feature matrix.

    Dataset columns (as published on Hugging Face):
      step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
      nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

    This loader is designed for QML experiments:
      - optionally balances (fraud vs non-fraud) because the raw dataset is highly imbalanced
      - encodes 'type' as one-hot or ordinal
      - drops high-cardinality IDs by default ('nameOrig', 'nameDest') unless include_names=True
      - can standardize and/or PCA-reduce features to a small dimension for quantum circuits

    Returns
    -------
    X: np.ndarray (float32), shape (n, d)
    y: np.ndarray (int64), shape (n,)
    feature_names: optional list[str]
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except Exception as e:
        raise ImportError(
            "This dataset loader requires the 'datasets' library. "
            "Install with: pip install datasets"
        ) from e

    type_encoding = str(type_encoding).lower().strip()
    if type_encoding not in ("onehot", "ordinal"):
        raise ValueError("type_encoding must be 'onehot' or 'ordinal'")

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    # -----------------------------
    # Load dataset
    # -----------------------------
    ds = load_dataset(dataset_id, split=split)

    # -----------------------------
    # Balance / sample
    # -----------------------------
    if balance:
        if n_fraud is None and n_legit is None:
            n_fraud = n_samples // 2
            n_legit = n_samples - n_fraud
        elif n_fraud is None:
            n_fraud = max(1, n_samples - int(n_legit))
        elif n_legit is None:
            n_legit = max(1, n_samples - int(n_fraud))

        n_fraud = int(n_fraud)
        n_legit = int(n_legit)

        # Filter is a full scan, but is deterministic and guarantees enough positives.
        fraud_ds = ds.filter(lambda ex: int(ex["isFraud"]) == 1, num_proc=num_proc)
        legit_ds = ds.filter(lambda ex: int(ex["isFraud"]) == 0, num_proc=num_proc)

        if n_fraud > len(fraud_ds):
            n_fraud = len(fraud_ds)
        if n_legit > len(legit_ds):
            n_legit = len(legit_ds)

        fraud_ds = fraud_ds.shuffle(seed=seed).select(range(n_fraud))
        legit_ds = legit_ds.shuffle(seed=seed + 1).select(range(n_legit))

        ds_small = concatenate_datasets([fraud_ds, legit_ds]).shuffle(seed=seed)
    else:
        # Simple random subset
        n = min(int(n_samples), len(ds))
        ds_small = ds.shuffle(seed=seed).select(range(n))

    # Convert the (small) subset to pandas for feature engineering
    df = ds_small.to_pandas()

    # -----------------------------
    # Labels
    # -----------------------------
    y = df["isFraud"].astype(np.int64).to_numpy()

    # -----------------------------
    # Numeric base features
    # -----------------------------
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    if include_isFlaggedFraud:
        num_cols.append("isFlaggedFraud")

    X_num = df[num_cols].astype(np.float32)

    # -----------------------------
    # Transaction type encoding
    # -----------------------------
    feats = [X_num]
    feat_names: List[str] = list(num_cols)

    if "type" in df.columns:
        t = df["type"].astype(str)
        if type_encoding == "onehot":
            cat = np.asarray(TYPE_CATEGORIES, dtype=object)
            tcat = np.asarray(t, dtype=object)
            # one-hot in a fixed, stable order
            onehot = np.stack([(tcat == c).astype(np.float32) for c in cat], axis=1)
            feats.append(onehot)
            feat_names.extend([f"type_{c}" for c in TYPE_CATEGORIES])
        else:
            mapping = {c: i for i, c in enumerate(TYPE_CATEGORIES)}
            ordinal = np.array([mapping.get(v, -1) for v in t], dtype=np.float32)[:, None]
            feats.append(ordinal)
            feat_names.append("type_ordinal")

    # -----------------------------
    # Optional hashed ID features (very lightweight; avoids one-hot explosion)
    # -----------------------------
    if include_names:
        def _hash01(s: Union[str, float, int]) -> float:
            # stable-ish hash -> [0,1)
            h = hash(str(s)) % 1_000_000
            return float(h) / 1_000_000.0

        name_orig = df.get("nameOrig", None)
        name_dest = df.get("nameDest", None)

        if name_orig is not None:
            v = np.array([_hash01(s) for s in name_orig], dtype=np.float32)[:, None]
            feats.append(v); feat_names.append("nameOrig_hash")
        if name_dest is not None:
            v = np.array([_hash01(s) for s in name_dest], dtype=np.float32)[:, None]
            feats.append(v); feat_names.append("nameDest_hash")

    X = np.concatenate([np.asarray(f, dtype=np.float32) for f in feats], axis=1)

    # -----------------------------
    # Optional standardization + PCA
    # -----------------------------
    if standardize or (pca_dim is not None):
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X).astype(np.float32)

    if pca_dim is not None:
        from sklearn.decomposition import PCA
        k = int(pca_dim)
        if k <= 0:
            raise ValueError("pca_dim must be > 0")
        X = PCA(n_components=k, random_state=seed).fit_transform(X).astype(np.float32)
        feat_names = [f"pca_{i}" for i in range(X.shape[1])]

    if return_feature_names:
        return X, y, feat_names
    return X, y
