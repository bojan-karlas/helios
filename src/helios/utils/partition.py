"""Partition-key helpers.

A component fans out over its ``partitions`` (e.g. ``[fold, size_mm, model]``).
These helpers expand a grid of partition values into the list of concrete key
dicts a component must run for.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any


def expand_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``{dim: [values]}`` -> list of ``{dim: value}`` dicts."""
    if not grid:
        return [{}]
    dims = list(grid)
    combos = itertools.product(*(list(grid[d]) for d in dims))
    return [dict(zip(dims, combo, strict=True)) for combo in combos]


def fold_values(n_folds: int | None) -> list[int | None]:
    """Fold coordinates to iterate over. ``None``/0/1 means a single fold-free run."""
    if not n_folds or n_folds <= 1:
        return [None]
    return list(range(n_folds))
