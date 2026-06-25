"""Resolve effective stage parameters with documented precedence.

Precedence (lowest to highest):

    stage signature defaults  (==  configs/default.yaml, generated)
        <  --config FILE  (a user override file, sparse)
        <  explicit CLI flags

Lists REPLACE (they are not merged). Only the keys for the requested
``verb -> noun`` stage are considered. The generated ``configs/default.yaml``
mirrors the signatures, so it is loaded as the base layer and a user
``--config`` only needs to specify the keys it changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from helios.config.introspect import build_default_config
from helios.data.catalog import _configs_dir


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data or {}


def load_stage_params(
    verb: str,
    noun: str,
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the effective parameter mapping for the ``verb noun`` stage.

    Starts from the committed ``configs/default.yaml`` (falling back to the live
    signatures if the file is missing), applies the ``verb.noun`` block of an
    optional ``--config`` file, then applies explicit ``overrides`` (CLI flags
    with a non-``None`` value). Lists replace.
    """
    default_file = _configs_dir() / "default.yaml"
    if default_file.exists():
        base = _load_yaml(default_file)
    else:
        base = build_default_config()
    params: dict[str, Any] = dict(base.get(verb, {}).get(noun, {}))

    if config_path is not None:
        user = _load_yaml(config_path)
        params.update(user.get(verb, {}).get(noun, {}))

    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    return params
