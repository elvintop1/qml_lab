from __future__ import annotations
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .vision_tabular import _maybe_limit, _maybe_pca, _standardize


def load_telco_churn(
    *, limit: Optional[int] = None,
    pca_dim: Optional[int] = None,
    seed: int = 0,
):
    """
    Loader for the Kaggle Telco Customer Churn dataset.

    It expects four CSV files in the same directory as this module:
      - X_train.csv
      - X_test.csv
      - y_train.csv
      - y_test.csv

    X_* must contain only numeric feature columns.
    y_* must contain a single column with integer labels (0 or 1).
    """
    base = Path(__file__).resolve().parent

    # Features
    Xtr = pd.read_csv(base / "X_train.csv").to_numpy(dtype=np.float32)
    Xte = pd.read_csv(base / "X_test.csv").to_numpy(dtype=np.float32)

    # Targets
    ytr_df = pd.read_csv(base / "y_train.csv")
    yte_df = pd.read_csv(base / "y_test.csv")

    if ytr_df.shape[1] != 1 or yte_df.shape[1] != 1:
        raise ValueError("y_train.csv and y_test.csv must have exactly one label column")

    ytr = ytr_df.iloc[:, 0].to_numpy(dtype=np.int64)
    yte = yte_df.iloc[:, 0].to_numpy(dtype=np.int64)

    # Combine train and test so paper_tables can do its own split
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)

    # Optional subsampling
    X, y = _maybe_limit(X, y, limit, seed)

    # Standardize and optionally apply PCA, same as other tabular datasets
    X = _maybe_pca(_standardize(X), pca_dim)

    return X, y
