"""HELIOS component stubs (helios.models.fusion.infer).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def fusion_infer(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """HELIOS Risk Inference.

    fold: reduce | mode: train, infer
    partitions: []
    consumes: cellular_risk_score
      path_concept_risk_score
      patch_morphology_risk_score
      staging_risk_score
      clinical_risk_score
      model_fusion
      cv_splits
    produces: helios_risk_score, survival_curve

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("fusion_infer is not implemented yet")


# --- /AUTOGEN:component-stubs ---
