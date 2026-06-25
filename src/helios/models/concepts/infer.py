"""HELIOS component stubs (helios.models.concepts.infer).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def concept_infer(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Pathology Concept Inference.

    fold: reduce | mode: train, infer
    partitions: []
    consumes: aligned_slide_embedding, whole_image_risk_score, model_concepts, cv_splits
    produces: path_concept_predictions, path_concept_risk_score

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("concept_infer is not implemented yet")


# --- /AUTOGEN:component-stubs ---
