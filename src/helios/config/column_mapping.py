"""Column mapping + data-dictionary binding.

A *column mapping* binds abstract pipeline inputs (targets, concepts, split key,
tiling fields) to concrete ``image_metadata`` column names via a YAML file, so
the same code runs on cohorts with different column names. The data dictionary
(``image_metadata.datadict.yaml``) is the canonical, versioned column contract
used for validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from helios.data.catalog import _configs_dir


def load_column_mapping(mapping_path: str | Path | None = None) -> dict[str, Any]:
    if mapping_path is None:
        mapping_path = _configs_dir() / "column_mapping" / "default.yaml"
    data = yaml.safe_load(Path(mapping_path).read_text())
    return data or {}


def load_datadict(datadict_path: str | Path | None = None) -> dict[str, Any]:
    if datadict_path is None:
        datadict_path = _configs_dir() / "schema" / "image_metadata.datadict.yaml"
    data = yaml.safe_load(Path(datadict_path).read_text())
    return data or {}


def resolve_column(mapping: dict[str, Any], name: str) -> Any:
    """Resolve a mapping name to its bound column(s); raises if unbound."""
    if name not in mapping:
        raise KeyError(f"{name!r} is not bound in the column mapping")
    return mapping[name]
