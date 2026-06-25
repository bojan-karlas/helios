"""HELIOS component stubs (helios.processing.tumor_patches).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def tumor_patch_extraction(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Tumor Cell Patch Extraction.

    fold: none | mode: train, infer
    partitions: []
    consumes: wsi, cells
    produces: tumor_cell_patches

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("tumor_patch_extraction is not implemented yet")


# --- /AUTOGEN:component-stubs ---
