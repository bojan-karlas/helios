"""HELIOS component stubs (helios.processing.tiling).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def tile_extraction(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Tile Extraction.

    fold: none | mode: train, infer
    partitions: ['size_mm', 'image_id']
    consumes: wsi, image_metadata
    produces: tile_metadata, tiles

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("tile_extraction is not implemented yet")


# --- /AUTOGEN:component-stubs ---
