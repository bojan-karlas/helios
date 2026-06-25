"""Component dispatch: component id -> stub callable.

Resolves the function for a component by importing its spec ``module`` and
getting the attribute named after the component id (stubs are generated 1:1 with
this convention). Keeps the pipeline wiring decoupled from import paths.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable

from helios.data.catalog import get_component

ComponentFn = Callable[..., None]


def get_component_fn(component_id: str) -> ComponentFn:
    comp = get_component(component_id)
    if not comp.module:
        raise ValueError(f"component {component_id!r} has no module declared in the spec")
    module = importlib.import_module(comp.module)
    try:
        return getattr(module, component_id)
    except AttributeError as exc:  # pragma: no cover - guards spec/stub drift
        raise AttributeError(
            f"component {component_id!r} has no function {component_id!r} in {comp.module!r}; "
            "regenerate stubs with scripts/gen_component_stubs.py"
        ) from exc
