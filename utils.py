import numpy as np
from typing import List, Tuple, Dict


def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)


def _canonical_edge(i: int, j: int) -> Tuple[int, int]:
    i = int(i)
    j = int(j)
    if i == j:
        raise ValueError("self-loop is not a valid two-qubit edge")
    return (i, j) if i < j else (j, i)


def edges_for(n: int, pattern: str) -> List[Tuple[int, int]]:
    """Return a duplicate-free undirected edge set.

    Older versions returned ``[(0, 1), (1, 0)]`` for a 2-qubit ring.  For CZ
    based encodings this silently cancels the entanglement because CZ^2 = I.
    This helper now canonicalizes edges and supports both ``linear`` and
    ``ring`` explicitly.
    """
    n = int(n)
    pattern = str(pattern or "none").lower().strip().replace("-", "_")
    if n < 2 or pattern in {"none", "empty", "off", "false", "0"}:
        return []
    if pattern in {"linear", "line", "chain"}:
        return [(i, i + 1) for i in range(n - 1)]
    if pattern == "ring":
        edges = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            edges.append((0, n - 1))
        return sorted({_canonical_edge(i, j) for i, j in edges})
    if pattern == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"Unknown entangler pattern: {pattern}")


def parse_kv(s: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not s:
        return d
    items = [t.strip() for t in s.split(",") if t.strip()]
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
            d[k.strip()] = v.strip()
        else:
            d[item] = "true"
    return d
