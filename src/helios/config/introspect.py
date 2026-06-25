"""Introspect stage signatures into a default configuration.

Code is the single source of truth: each stage in :data:`helios.stages.STAGES`
declares its tunable parameters as typed keyword arguments with defaults. This
module reads those signatures (so ``helios <verb> <noun> --help`` and the config
keys cannot drift) and builds the nested mapping that ``configs/default.yaml``
serializes. Run ``make configs`` to regenerate the committed file.

The config mirrors the CLI shape: ``{verb: {noun: {param: default}}}``.

Framework seams are runtime arguments, not configuration, and are excluded:
``dataset_root``, ``roots``, ``image_ids``, ``force``, ``progress``.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from helios.stages import STAGES

#: Parameters that are runtime seams, never emitted to config.
RUNTIME_PARAMS = frozenset(
    {"datasets", "output_root", "roots", "image_ids", "force", "progress"}
)


def stage_params(fn: Callable[..., Any]) -> dict[str, Any]:
    """Return the tunable ``name -> default`` mapping for one stage function."""
    params: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in RUNTIME_PARAMS:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        params[name] = param.default
    return params


def build_default_config() -> dict[str, dict[str, dict[str, Any]]]:
    """Build the full default config mapping, keyed by ``verb -> noun``."""
    return {
        verb: {noun: stage_params(fn) for noun, fn in nouns.items()}
        for verb, nouns in STAGES.items()
    }
