"""HELIOS component stubs (helios.data.cv_splits).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def cv_splits_generation(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """CV Splits Generation.

    fold: none | mode: train
    partitions: []
    consumes: image_metadata
    produces: cv_splits

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("cv_splits_generation is not implemented yet")


# --- /AUTOGEN:component-stubs ---
