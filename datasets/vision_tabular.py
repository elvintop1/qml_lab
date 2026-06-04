from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from sklearn.datasets import (
    fetch_openml,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_iris,
)
from sklearn.decomposition import PCA

def _standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd

def _maybe_limit(X: np.ndarray, y: np.ndarray, limit: Optional[int], seed: int):
    if limit is not None and X.shape[0] > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=limit, replace=False)
        return X[idx], y[idx]
    return X, y

def _maybe_pca(X: np.ndarray, k: Optional[int]) -> np.ndarray:
    if k is None or k >= X.shape[1]:
        return X
    return PCA(n_components=k, random_state=0).fit_transform(X)

def load_vision_tabular(name: str, *, limit: Optional[int] = None,
                        pca_dim: Optional[int] = None, seed: int = 0):
    name = name.lower()

    if name == "mnist":
        try:
            ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X, y = ds.data, ds.target.astype(np.int64)
        except Exception as e:
            print(f"[offline] MNIST via OpenML failed ({e.__class__.__name__}: {e}). Trying Keras MNIST...")
            try:
                from tensorflow.keras.datasets import mnist as k_mnist  # type: ignore
                (Xtr, ytr), (Xte, yte) = k_mnist.load_data()
                X = np.concatenate([Xtr, Xte], axis=0).reshape(-1, 28 * 28).astype(np.float32)
                y = np.concatenate([ytr, yte], axis=0).astype(np.int64)
            except Exception as e2:
                print(f"[offline] Keras MNIST failed ({e2.__class__.__name__}). Falling back to sklearn.digits.")
                d = load_digits()
                X, y = d.data.astype(np.float32), d.target.astype(np.int64)

        X, y = _maybe_limit(X, y, limit, seed)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name == "fashion_mnist":
        try:
            ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
            X, y = ds.data, ds.target.astype(np.int64)
        except Exception as e:
            print(f"[offline] Fashion-MNIST via OpenML failed ({e.__class__.__name__}: {e}). Trying Keras Fashion-MNIST...")
            try:
                from tensorflow.keras.datasets import fashion_mnist as k_fmnist # type: ignore
                (Xtr, ytr), (Xte, yte) = k_fmnist.load_data()
                X = np.concatenate([Xtr, Xte], axis=0).reshape(-1, 28 * 28).astype(np.float32)
                y = np.concatenate([ytr, yte], axis=0).astype(np.int64)
            except Exception as e2:
                print(f"[offline] Keras Fashion-MNIST failed ({e2.__class__.__name__}). Falling back to sklearn.digits.")
                d = load_digits()
                X, y = d.data.astype(np.float32), d.target.astype(np.int64)

        X, y = _maybe_limit(X, y, limit, seed)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name in {"cifar10", "cifar_10", "cifar-10"}:
        ds = fetch_openml("cifar_10", version="active", as_frame=False, parser="auto")
        X = np.asarray(ds.data, dtype=np.float32)
        labels, y = np.unique(np.asarray(ds.target), return_inverse=True)
        y = y.astype(np.int64)
        X, y = _maybe_limit(X, y, limit, seed)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name == "digits":
        d = load_digits()
        X, y = d.data.astype(np.float32), d.target.astype(np.int64)
        X, y = _maybe_limit(X, y, limit, seed)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name == "wine":
        d = load_wine()
        X, y = d.data.astype(np.float32), d.target.astype(np.int64)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data.astype(np.float32), d.target.astype(np.int64)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    if name == "iris":
        d = load_iris()
        X, y = d.data.astype(np.float32), d.target.astype(np.int64)
        X = _maybe_pca(_standardize(X), pca_dim)
        return X, y

    raise ValueError(f"Unknown dataset '{name}'")

def load_dataset(name: str, **kw):
    return load_vision_tabular(name, **kw)
