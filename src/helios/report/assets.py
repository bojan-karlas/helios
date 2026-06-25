"""HELIOS component stubs (helios.report.assets).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def assets_render(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Render Assets.

    fold: reduce | mode: infer
    partitions: []
    consumes: case_report
      tile_metadata
      tile_clusters
      patch_attention
      cells
      thumbnail
      cv_splits
    produces: case_assets

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("assets_render is not implemented yet")


# --- /AUTOGEN:component-stubs ---
