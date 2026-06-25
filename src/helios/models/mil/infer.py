"""HELIOS component stubs (helios.models.mil.infer).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def mil_inference(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """A-MIL Inference.

    fold: map | mode: train, infer
    partitions: ['fold']
    consumes: tile_features, tile_background, model_mil
    produces: aligned_patch_embeddings
      patch_attention
      aligned_slide_embedding
      whole_image_risk_score

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("mil_inference is not implemented yet")


# --- /AUTOGEN:component-stubs ---
