"""Encoding registry.

This repo uses a small registry to map a user-facing encoding name (string)
to an Encoding class.

Phase-2 thesis runners need to support multiple spelling variants such as:
  - denseangle / dense_angle / dense-angle
  - trainablekernel / trainable_kernel

To make CLI usage robust, we canonicalize encoding names by:
  - lowercasing
  - stripping whitespace
  - converting '-' to '_'
  - removing '_' entirely
"""

from __future__ import annotations

from typing import Dict, List


ENCODINGS: Dict[str, type] = {}


def _canon(name: str) -> str:
    s = str(name).lower().strip()
    s = s.replace("-", "_")
    s = s.replace(" ", "")
    s = s.replace("_", "")
    return s


def register_encoding(name: str):
    """Decorator to register an Encoding class under a name."""

    def deco(cls):
        key = _canon(name)
        # NOTE:
        # We canonicalize names by removing '_' and '-' so that CLI usage is
        # robust (e.g., "trainable_kernel" == "trainablekernel").
        #
        # Several encodings in the thesis runner deliberately use multiple
        # decorators to register aliases on the same class. Those aliases may
        # canonicalize to the same key. In that case, treat registration as
        # idempotent rather than raising.
        if key in ENCODINGS:
            if ENCODINGS[key] is cls:
                return cls
            raise ValueError(
                f"Encoding name '{name}' (canon='{key}') already registered."
            )

        ENCODINGS[key] = cls
        # Keep a readable, user-facing encoding name on the class.
        if not getattr(cls, "encoding_name", None):
            cls.encoding_name = key
        return cls

    return deco


def get_encoding_cls(name: str):
    """Resolve a user-provided encoding name to the registered class."""
    key = _canon(name)
    if key not in ENCODINGS:
        raise KeyError(f"Unknown encoding '{name}'. Available: {sorted(ENCODINGS.keys())}")
    return ENCODINGS[key]


def list_encodings() -> List[str]:
    return sorted(ENCODINGS.keys())
