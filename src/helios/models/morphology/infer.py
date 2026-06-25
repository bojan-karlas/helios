"""HELIOS component stubs (helios.models.morphology.infer).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def morphology_infer(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Morphology Clustering Risk Inference.

    fold: reduce | mode: train, infer
    partitions: []
    consumes: aligned_patch_embeddings, patch_attention, model_morphology, cv_splits
    produces: tile_clusters, patch_morphology_presence, patch_morphology_risk_score

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("morphology_infer is not implemented yet")


# --- /AUTOGEN:component-stubs ---
