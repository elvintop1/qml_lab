from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PaperProblem:
    name: str
    dim: int
    classes: int
    default_train: int
    default_test: int


PAPER_PROBLEMS: Dict[str, PaperProblem] = {
    "circle": PaperProblem("circle", dim=2, classes=2, default_train=200, default_test=4000),
    "3_circles": PaperProblem("3_circles", dim=2, classes=4, default_train=200, default_test=4000),
    "wavy_lines": PaperProblem("wavy_lines", dim=2, classes=4, default_train=200, default_test=4000),
    "squares": PaperProblem("squares", dim=2, classes=4, default_train=200, default_test=4000),
    "non_convex": PaperProblem("non_convex", dim=2, classes=2, default_train=200, default_test=4000),
    "crown": PaperProblem("crown", dim=2, classes=2, default_train=200, default_test=4000),
    "tricrown": PaperProblem("tricrown", dim=2, classes=3, default_train=200, default_test=4000),
    "sphere": PaperProblem("sphere", dim=3, classes=2, default_train=500, default_test=4000),
    "hypersphere": PaperProblem("hypersphere", dim=4, classes=2, default_train=1000, default_test=4000),
}

_ALIASES = {
    "3 circles": "3_circles",
    "3-circles": "3_circles",
    "three_circles": "3_circles",
    "wavy lines": "wavy_lines",
    "wavy-lines": "wavy_lines",
    "non convex": "non_convex",
    "non-convex": "non_convex",
}


def canonical_problem_name(problem: str) -> str:
    key = str(problem).strip().lower()
    key = _ALIASES.get(key, key)
    key = key.replace("-", "_").replace(" ", "_")
    if key not in PAPER_PROBLEMS:
        valid = ", ".join(PAPER_PROBLEMS)
        raise ValueError(f"Unknown paper problem '{problem}'. Valid problems: {valid}")
    return key


def default_split(problem: str) -> Tuple[int, int]:
    spec = PAPER_PROBLEMS[canonical_problem_name(problem)]
    return spec.default_train, spec.default_test


def _sample_uniform(rng: np.random.Generator, samples: int, dim: int) -> np.ndarray:
    return 2.0 * rng.random((int(samples), int(dim))) - 1.0


def _circle(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    y = (np.linalg.norm(x, axis=1) < np.sqrt(2.0 / np.pi)).astype(int)
    return x, y


def _3_circles(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    centers = np.array([[-1.0, 1.0], [1.0, 0.0], [-0.5, -0.5]])
    radii = np.array([1.0, np.sqrt(6.0 / np.pi - 1.0), 0.5])
    y = np.zeros(samples, dtype=int)
    for idx, (center, radius) in enumerate(zip(centers, radii), start=1):
        y[np.linalg.norm(x - center, axis=1) < radius] = idx
    return x, y


def _wavy_lines(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    f1 = x[:, 0] + np.sin(np.pi * x[:, 0])
    f2 = -x[:, 0] + np.sin(np.pi * x[:, 0])
    y = np.zeros(samples, dtype=int)
    y[(x[:, 1] < f1) & (x[:, 1] > f2)] = 1
    y[(x[:, 1] > f1) & (x[:, 1] < f2)] = 2
    y[(x[:, 1] > f1) & (x[:, 1] > f2)] = 3
    return x, y


def _squares(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    y = np.zeros(samples, dtype=int)
    y[(x[:, 0] < 0.0) & (x[:, 1] > 0.0)] = 1
    y[(x[:, 0] > 0.0) & (x[:, 1] < 0.0)] = 2
    y[(x[:, 0] > 0.0) & (x[:, 1] > 0.0)] = 3
    return x, y


def _non_convex(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    boundary = -2.0 * x[:, 0] + 1.5 * np.sin(np.pi * x[:, 0])
    y = (x[:, 1] > boundary).astype(int)
    return x, y


def _crown(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    norm = np.linalg.norm(x, axis=1)
    outer = np.sqrt(0.8)
    inner = np.sqrt(0.8 - 2.0 / np.pi)
    y = ((norm < outer) & (norm > inner)).astype(int)
    return x, y


def _tricrown(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 2)
    norm = np.linalg.norm(x, axis=1)
    radii = [np.sqrt(0.8 - 2.0 / np.pi), np.sqrt(0.8)]
    y = np.zeros(samples, dtype=int)
    for idx, radius in enumerate(radii, start=1):
        y[norm > radius] = idx
    return x, y


def _sphere(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 3)
    y = (np.linalg.norm(x, axis=1) < (3.0 / np.pi) ** (1.0 / 3.0)).astype(int)
    return x, y


def _hypersphere(rng: np.random.Generator, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sample_uniform(rng, samples, 4)
    y = (np.linalg.norm(x, axis=1) < np.sqrt(2.0 / np.pi)).astype(int)
    return x, y


_GENERATORS: Dict[str, Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]] = {
    "circle": _circle,
    "3_circles": _3_circles,
    "wavy_lines": _wavy_lines,
    "squares": _squares,
    "non_convex": _non_convex,
    "crown": _crown,
    "tricrown": _tricrown,
    "sphere": _sphere,
    "hypersphere": _hypersphere,
}


def generate_paper_problem(
    problem: str,
    samples: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate one of the datasets used in arXiv:1907.02085.

    Points are sampled directly in [-1, 1]^d. No PCA or feature reduction is
    applied; the problem dimension is part of the synthetic task definition.
    """
    key = canonical_problem_name(problem)
    local_rng = rng if rng is not None else np.random.default_rng(seed)
    x, y = _GENERATORS[key](local_rng, int(samples))
    spec = PAPER_PROBLEMS[key]
    meta = {
        "name": spec.name,
        "dim": spec.dim,
        "classes": spec.classes,
        "source": "arXiv:1907.02085v3",
        "pca": False,
        "domain": "[-1, 1]^d",
    }
    return x.astype(float, copy=False), y.astype(int, copy=False), meta


def make_paper_1907_split(
    problem: str,
    *,
    train_samples: int | None = None,
    test_samples: int | None = None,
    seed: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create independent train/test samples following the paper protocol."""
    key = canonical_problem_name(problem)
    default_train, default_test = default_split(key)
    n_train = int(train_samples if train_samples is not None else default_train)
    n_test = int(test_samples if test_samples is not None else default_test)
    rng = np.random.default_rng(int(seed))
    x_train, y_train, meta = generate_paper_problem(key, n_train, rng=rng)
    x_test, y_test, _ = generate_paper_problem(key, n_test, rng=rng)
    meta.update({"train_samples": n_train, "test_samples": n_test, "seed": int(seed)})
    return x_train, y_train, x_test, y_test, meta


def load_paper_1907(
    problem: str = "circle",
    *,
    samples: int | None = None,
    limit: int | None = None,
    seed: int = 30,
    pca_dim: int | None = None,
    **_: object,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Dataset-loader wrapper for experiments/run.py compatibility."""
    if pca_dim is not None:
        raise ValueError("Paper 1907 datasets are defined without PCA; omit --pca-dim.")
    key = canonical_problem_name(problem)
    default_train, default_test = default_split(key)
    total = int(samples if samples is not None else (limit if limit is not None else default_train + default_test))
    return generate_paper_problem(key, total, seed=int(seed))
