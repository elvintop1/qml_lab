from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine, make_classification
from sklearn.decomposition import PCA


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-12
    return (x - mu) / sd


def _stratified_limit(x: np.ndarray, y: np.ndarray, limit: int | None, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if limit is None or int(limit) <= 0 or int(limit) >= len(y):
        return x, y
    rng = np.random.default_rng(int(seed))
    labels, counts = np.unique(y, return_counts=True)
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
    chosen = []
    for label, count in zip(labels, target):
        pool = np.where(y == label)[0]
        chosen.append(rng.choice(pool, size=min(int(count), len(pool)), replace=False))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return x[idx], y[idx]


def _binary_first_two(x: np.ndarray, y: np.ndarray, labels: Tuple[int, int] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    chosen = np.asarray(labels if labels is not None else np.unique(y)[:2], dtype=int)
    mask = np.isin(y, chosen)
    mapping = {int(old): new for new, old in enumerate(chosen.tolist())}
    y_bin = np.asarray([mapping[int(v)] for v in y[mask]], dtype=int)
    return np.asarray(x)[mask], y_bin


def _load_iris_4(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = load_iris()
    x, y = _binary_first_two(data.data, data.target, labels=(0, 1))
    return _standardize(x), y, {"source": "sklearn.load_iris", "features": 4, "labels": [0, 1]}


def _load_wine_13(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = load_wine()
    x, y = _binary_first_two(data.data, data.target, labels=(0, 1))
    return _standardize(x), y, {"source": "sklearn.load_wine", "features": 13, "labels": [0, 1]}


def _load_breast_cancer_30(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = load_breast_cancer()
    x = _standardize(data.data)
    y = np.asarray(data.target, dtype=int)
    return x, y, {"source": "sklearn.load_breast_cancer", "features": 30, "labels": [0, 1]}


def _load_digits_32var(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = load_digits()
    x, y = _binary_first_two(data.data, data.target, labels=(3, 6))
    x = _standardize(x)
    top = np.argsort(np.var(x, axis=0))[-32:]
    top = np.sort(top)
    return x[:, top], y, {"source": "sklearn.load_digits classes 3 vs 6, top variance pixels", "features": 32, "labels": [3, 6]}


def _load_mnist_36_pca(seed: int, limit: int | None, n_components: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    from qml_lab.datasets.vision_tabular import load_vision_tabular

    x_all, y_all = load_vision_tabular("mnist", limit=None, pca_dim=None, seed=int(seed))
    x, y = _binary_first_two(x_all, y_all, labels=(3, 6))
    x, y = _stratified_limit(x, y, limit, int(seed))

    n_components = int(n_components)
    if x.shape[0] <= n_components:
        raise ValueError(f"mnist_36_{n_components}pca needs more than {n_components} selected samples for PCA.")
    x = PCA(n_components=n_components, random_state=int(seed)).fit_transform(_standardize(x))
    return _standardize(x), y, {
        "source": f"MNIST digit 3 vs 6 via OpenML/Keras fallback, PCA({n_components})",
        "features": n_components,
        "labels": [3, 6],
        "pca": True,
    }


def _load_mnist_36_16pca(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    return _load_mnist_36_pca(seed, limit, 16)


def _load_mnist_36_32pca(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    return _load_mnist_36_pca(seed, limit, 32)


def _load_cifar10_raw() -> Tuple[np.ndarray, np.ndarray, str]:
    def load_toronto_archive() -> Tuple[np.ndarray, np.ndarray, str]:
        import hashlib
        import pickle
        import tarfile
        import urllib.request
        from pathlib import Path

        from sklearn.datasets import get_data_home

        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        expected_md5 = "c58f30108f718f92721af3b95e74349a"
        cache_dir = Path(get_data_home()) / "cifar10_toronto"
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "cifar-10-python.tar.gz"

        if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest() != expected_md5:
            urllib.request.urlretrieve(url, archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != expected_md5:
            raise RuntimeError("Downloaded CIFAR-10 Toronto archive failed md5 validation.")

        xs = []
        ys = []
        with tarfile.open(archive, "r:gz") as tf:
            for member_name in [
                "cifar-10-batches-py/data_batch_1",
                "cifar-10-batches-py/data_batch_2",
                "cifar-10-batches-py/data_batch_3",
                "cifar-10-batches-py/data_batch_4",
                "cifar-10-batches-py/data_batch_5",
                "cifar-10-batches-py/test_batch",
            ]:
                member = tf.getmember(member_name)
                fh = tf.extractfile(member)
                if fh is None:
                    raise RuntimeError(f"Could not read {member_name} from CIFAR-10 archive.")
                batch = pickle.load(fh, encoding="latin1")
                xs.append(np.asarray(batch["data"], dtype=np.float32))
                ys.append(np.asarray(batch["labels"], dtype=int))
        return np.vstack(xs), np.concatenate(ys), "Toronto CIFAR-10 python archive"

    try:
        from pathlib import Path

        from sklearn.datasets import get_data_home

        cached_archive = Path(get_data_home()) / "cifar10_toronto" / "cifar-10-python.tar.gz"
        if cached_archive.exists():
            return load_toronto_archive()
    except Exception:
        pass

    try:
        data = fetch_openml("cifar_10", version="active", as_frame=False, parser="auto")
        x = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(data.target).astype(int)
        return x, y, "OpenML cifar_10"
    except Exception as exc:
        print(f"[offline] CIFAR-10 via OpenML failed ({exc.__class__.__name__}: {exc}). Trying Keras CIFAR-10...")
        try:
            from tensorflow.keras.datasets import cifar10 as k_cifar10  # type: ignore

            (xtr, ytr), (xte, yte) = k_cifar10.load_data()
            x = np.concatenate([xtr, xte], axis=0).reshape(-1, 32 * 32 * 3).astype(np.float32)
            y = np.concatenate([ytr, yte], axis=0).ravel().astype(int)
            return x, y, "Keras CIFAR-10"
        except Exception as exc2:
            print(f"[offline] Keras CIFAR-10 failed ({exc2.__class__.__name__}). Trying Toronto CIFAR-10 archive...")
            return load_toronto_archive()


def _load_cifar10_01_16pca(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    x_all, y_all, source = _load_cifar10_raw()
    x, y = _binary_first_two(x_all, y_all, labels=(0, 1))
    x, y = _stratified_limit(x, y, limit, int(seed))

    if x.shape[0] <= 16:
        raise ValueError("cifar10_01_16pca needs more than 16 selected samples for PCA.")
    x = PCA(n_components=16, random_state=int(seed)).fit_transform(_standardize(x))
    return _standardize(x), y, {
        "source": f"{source} class 0 vs 1, PCA(16)",
        "features": 16,
        "labels": [0, 1],
        "pca": True,
    }


def _load_synthetic(
    n_features: int,
    seed: int,
    *,
    n_samples: int = 1200,
    limit: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    actual_samples = int(n_samples)
    if limit is not None and int(limit) > 0:
        actual_samples = min(actual_samples, max(int(limit), 2))
    informative = max(2, min(n_features, n_features // 2))
    redundant = max(0, min(n_features - informative, n_features // 4))
    x, y = make_classification(
        n_samples=int(actual_samples),
        n_features=int(n_features),
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.02,
        random_state=int(seed),
    )
    return _standardize(x), y.astype(int), {
        "source": f"sklearn.make_classification({n_features} features)",
        "features": int(n_features),
        "labels": [0, 1],
        "full_samples": int(n_samples),
    }


def _load_poker_hand_1m(seed: int, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    try:
        from ucimlrepo import fetch_ucirepo

        poker = fetch_ucirepo(id=158)
        x = poker.data.features.to_numpy(dtype=np.float32)
        y_raw = poker.data.targets.to_numpy().ravel().astype(int)
    except Exception:
        import pandas as pd

        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker"
        names = [f"S{i}" if j % 2 == 0 else f"C{i}" for i in range(1, 6) for j in range(2)]
        cols = names + ["CLASS"]
        train = pd.read_csv(f"{base}/poker-hand-training-true.data", header=None, names=cols)
        test = pd.read_csv(f"{base}/poker-hand-testing.data", header=None, names=cols)
        data = pd.concat([train, test], ignore_index=True)
        x = data[names].to_numpy(dtype=np.float32)
        y_raw = data["CLASS"].to_numpy(dtype=int)

    # Binary task: no recognized hand vs one-pair hand. These two classes hold
    # most rows and keep the quantum classifiers' binary interface intact.
    x, y = _binary_first_two(x, y_raw, labels=(0, 1))
    return _standardize(x), y, {
        "source": "UCI Poker Hand, id=158, class 0 vs 1",
        "features": 10,
        "labels": [0, 1],
        "full_samples": 1_025_010,
    }


def _synthetic_loader(
    n_features: int,
    n_samples: int = 1200,
) -> Callable[[int, int | None], Tuple[np.ndarray, np.ndarray, dict]]:
    return lambda seed, limit=None: _load_synthetic(n_features, seed, n_samples=n_samples, limit=limit)


FEATURE_SWEEP_LOADERS: Dict[str, Callable[[int, int | None], Tuple[np.ndarray, np.ndarray, dict]]] = {
    "iris_4": _load_iris_4,
    "synthetic_8": _synthetic_loader(8),
    "wine_13": _load_wine_13,
    "synthetic_16": _synthetic_loader(16),
    "breast_cancer_30": _load_breast_cancer_30,
    "digits_32var": _load_digits_32var,
    "mnist_36_16pca": _load_mnist_36_16pca,
    "mnist_36_32pca": _load_mnist_36_32pca,
    "cifar10_01_16pca": _load_cifar10_01_16pca,
    "synthetic_32": _synthetic_loader(32),
    "synthetic_1m_8": _synthetic_loader(8, 1_000_000),
    "synthetic_1m_32": _synthetic_loader(32, 1_000_000),
    "poker_hand_1m": _load_poker_hand_1m,
}


DEFAULT_FEATURE_SWEEP = [
    "iris_4",
    "synthetic_8",
    "wine_13",
    "synthetic_16",
    "breast_cancer_30",
    "digits_32var",
    "synthetic_32",
]


def load_feature_sweep_dataset(name: str, *, seed: int = 42, limit: int | None = None, **_: object):
    key = str(name).strip().lower()
    if key not in FEATURE_SWEEP_LOADERS:
        valid = ", ".join(FEATURE_SWEEP_LOADERS)
        raise ValueError(f"Unknown feature-sweep dataset '{name}'. Valid: {valid}")
    x, y, meta = FEATURE_SWEEP_LOADERS[key](int(seed), limit)
    x, y = _stratified_limit(x, y, limit, int(seed))
    meta = dict(meta)
    meta.update({"name": key, "pca": False, "samples": int(len(y))})
    return x.astype(np.float32, copy=False), y.astype(int, copy=False), meta
