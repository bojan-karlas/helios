"""HELIOS component stubs (helios.processing.normalization).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def stain_normalization(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Stain Normalization / Augmentation.

    fold: none | mode: train, infer
    partitions: ['target', 'size_mm', 'image_id']
    consumes: tiles, model_cyclegan
    produces: tile_normalization

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("stain_normalization is not implemented yet")


# --- /AUTOGEN:component-stubs ---
