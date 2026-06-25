"""Config loading + shallow layering.

Loads YAML into :class:`~helios.config.models.RunConfig`, applying a simple
override precedence: ``base.yaml`` (shipped defaults) -> user config -> explicit
overrides dict (e.g. from the CLI). Nested dicts are merged shallowly per
top-level key, which is enough for the scaffold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from helios.config.models import RunConfig
from helios.data.catalog import _configs_dir  # reuse the configs/ locator


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    return data or {}


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> RunConfig:
    base_path = _configs_dir() / "base.yaml"
    merged: dict = _read_yaml(base_path) if base_path.exists() else {}

    if config_path is not None:
        merged = _deep_merge(merged, _read_yaml(config_path))
    if overrides:
        merged = _deep_merge(merged, overrides)

    return RunConfig.model_validate(merged)
