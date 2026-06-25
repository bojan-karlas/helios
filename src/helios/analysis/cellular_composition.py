"""HELIOS component stubs (helios.analysis.cellular_composition).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def tile_cell_aggregation(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Per-Tile Cell Aggregation.

    fold: none | mode: train, infer
    partitions: []
    consumes: cells, mitotic_cells, tile_metadata
    produces: tile_cell_counts

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("tile_cell_aggregation is not implemented yet")


def cellular_features_compute(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Cellular Composition Feature Computation.

    fold: none | mode: train, infer
    partitions: []
    consumes: tile_cell_counts
    produces: cellular_features

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("cellular_features_compute is not implemented yet")


# --- /AUTOGEN:component-stubs ---
