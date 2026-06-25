"""HELIOS component stubs (helios.processing.features).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def feature_extraction(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Foundation Feature Extraction.

    fold: none | mode: train, infer
    partitions: ['size_mm', 'model', 'image_id']
    consumes: tiles
    produces: tile_features

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("feature_extraction is not implemented yet")


def feature_extraction_normalized(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Foundation Feature Extraction (Normalized).

    fold: none | mode: train, infer
    partitions: ['target', 'size_mm', 'model', 'image_id']
    consumes: tile_normalization
    produces: tile_normalization_features

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("feature_extraction_normalized is not implemented yet")


# --- /AUTOGEN:component-stubs ---
