"""HELIOS component stubs (helios.models.mil.reduce).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext

# --- AUTOGEN:component-stubs (do not edit between markers) ---

def whole_image_risk_reduce(ctx: ComponentContext, keys: dict[str, Any]) -> None:
    """Whole-Image Risk Reduction.

    fold: reduce | mode: train, infer
    partitions: []
    consumes: whole_image_risk_score, cv_splits
    produces: whole_image_risk_score_ensemble

    ``keys`` holds this invocation's partition coordinates.
    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via
    ``ctx.store.write(<id>, obj, **keys)``.
    """
    raise NotImplementedError("whole_image_risk_reduce is not implemented yet")


# --- /AUTOGEN:component-stubs ---
