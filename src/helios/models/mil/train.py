"""HELIOS component stubs (helios.models.mil.train).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def mil_training(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """A-MIL Training.

    fold: map | mode: train
    partitions: ['fold', 'size_mm', 'model']
    consumes: tile_features
      tile_normalization_features
      tile_background
      image_metadata
      cv_splits
    produces: model_mil

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("mil_training is not implemented yet")


# --- /AUTOGEN:component-stubs ---
