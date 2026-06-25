"""HELIOS component stubs (helios.report.summary).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def cohort_summarize(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Cohort Summary View.

    fold: none | mode: infer
    partitions: []
    consumes: helios_risk_score
      cellular_risk_score
      path_concept_risk_score
      patch_morphology_risk_score
      staging_risk_score
      clinical_risk_score
      whole_image_risk_score_ensemble
      cellular_features
      path_concept_predictions
      patch_morphology_presence
      survival_curve
      image_metadata
    produces: cohort_summary

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("cohort_summarize is not implemented yet")


# --- /AUTOGEN:component-stubs ---
