from __future__ import annotations

from typing import Any, Callable, Dict

from .vision_tabular import load_dataset as load_vision_tabular

_DATASETS: Dict[str, Callable[..., Any]] = {}


def register_dataset(name: str, loader: Callable[..., Any]) -> None:
    """Register a dataset loader under a string key."""
    _DATASETS[str(name)] = loader


def get_dataset(name: str) -> Callable[..., Any]:
    try:
        return _DATASETS[str(name)]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Registered: {list(_DATASETS)}")


def load_dataset(name: str, **kw):
    """Convenience wrapper: load a dataset by name (matches experiments/run.py)."""
    return get_dataset(name)(**kw)


# -----------------------------
# Built-in datasets
# -----------------------------
register_dataset("mnist", lambda **kw: load_vision_tabular("mnist", **kw))
register_dataset("cifar10", lambda **kw: load_vision_tabular("cifar10", **kw))
register_dataset("cifar_10", lambda **kw: load_vision_tabular("cifar10", **kw))
register_dataset("fashion_mnist", lambda **kw: load_vision_tabular("fashion_mnist", **kw))
register_dataset("digits", lambda **kw: load_vision_tabular("digits", **kw))
register_dataset("wine", lambda **kw: load_vision_tabular("wine", **kw))
register_dataset("breast_cancer", lambda **kw: load_vision_tabular("breast_cancer", **kw))
register_dataset("iris", lambda **kw: load_vision_tabular("iris", **kw))

from .paper_1907 import load_paper_1907

for _paper_problem in [
    "circle",
    "3_circles",
    "wavy_lines",
    "squares",
    "non_convex",
    "crown",
    "tricrown",
    "sphere",
    "hypersphere",
]:
    register_dataset(f"paper_1907_{_paper_problem}", lambda _p=_paper_problem, **kw: load_paper_1907(_p, **kw))
    register_dataset(f"paper_{_paper_problem}", lambda _p=_paper_problem, **kw: load_paper_1907(_p, **kw))

from .feature_sweep import FEATURE_SWEEP_LOADERS, load_feature_sweep_dataset

for _feature_dataset in FEATURE_SWEEP_LOADERS:
    register_dataset(_feature_dataset, lambda _d=_feature_dataset, **kw: load_feature_sweep_dataset(_d, **kw))

from .kaggle_telco import load_telco_churn
register_dataset("telco_churn", lambda **kw: load_telco_churn(**kw))

# -----------------------------
# Phase-2 / Hugging Face datasets
# -----------------------------
try:
    from .hf_synthetic_fraud import load_hf_synthetic_fraud
    register_dataset("hf_synthetic_fraud", lambda **kw: load_hf_synthetic_fraud(**kw))
except Exception:
    # Optional dependency (datasets) not installed in some environments.
    pass

# -----------------------------
# UCI datasets (large / time-series)
# -----------------------------
try:
    from .uci_elder_gas_pos import load_uci_elder_gas_pos
    register_dataset("uci_elder_gas_pos", lambda **kw: load_uci_elder_gas_pos(**kw))
except Exception:
    # Optional dependency (ucimlrepo) not installed in some environments.
    pass
