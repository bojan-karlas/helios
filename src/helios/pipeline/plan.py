"""Hand-wired pipeline plans (approach B).

``PIPELINE_ORDER`` is the single canonical, topologically-valid ordering of all
components (kept in sync with the spec phases). The train/infer plans are this
order filtered by each component's declared ``mode``. When we later build a
generic engine, this hand-wired order is what it must reproduce.
"""
from __future__ import annotations

from helios.config.models import RunConfig
from helios.data.catalog import get_component, load_components
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver
from helios.pipeline.context import ComponentContext
from helios.pipeline.runner import StageResult, run_plan
from helios.progress.logging import configure_logging
from helios.progress.reporter import ProgressReporter

PIPELINE_ORDER: list[str] = [
    "cv_splits_generation",
    "thumbnail_extraction",
    "tile_extraction",
    "background_removal",
    "stain_normalization",
    "feature_extraction",
    "feature_extraction_normalized",
    "cell_segmentation",
    "tumor_patch_extraction",
    "mitotic_classification",
    "tile_cell_aggregation",
    "cyclegan_training",
    "mil_training",
    "mil_inference",
    "whole_image_risk_reduce",
    "cellular_features_compute",
    "cellular_risk_training",
    "cellular_risk_infer",
    "concept_models_training",
    "concept_infer",
    "morphology_training",
    "morphology_infer",
    "staging_risk_training",
    "staging_risk_infer",
    "clinical_risk_training",
    "clinical_risk_infer",
    "fusion_training",
    "fusion_infer",
    "case_report_build",
    "assets_render",
    "markdown_render",
    "cohort_summarize",
]


def _plan_for_mode(mode: str) -> list[str]:
    return [cid for cid in PIPELINE_ORDER if mode in (get_component(cid).mode or [])]


def train_plan() -> list[str]:
    return _plan_for_mode("train")


def infer_plan() -> list[str]:
    return _plan_for_mode("infer")


def build_context(config: RunConfig) -> ComponentContext:
    roots = config.paths.to_roots()
    resolver = Resolver(roots)
    store = ArtifactStore(resolver)
    output_root = roots._role("output")
    reporter = ProgressReporter(output_root)
    logger = configure_logging(output_root)
    from helios.config.roles import load_roles

    roles = load_roles(config.roles)
    return ComponentContext(
        config=config,
        store=store,
        resolver=resolver,
        reporter=reporter,
        logger=logger,
        roles=roles,
    )


def _validate_order() -> None:
    """Guard: the hand-wired order must list every spec component exactly once."""
    spec_ids = set(load_components())
    order_ids = set(PIPELINE_ORDER)
    missing = spec_ids - order_ids
    extra = order_ids - spec_ids
    if missing or extra:
        raise RuntimeError(f"PIPELINE_ORDER drift: missing={sorted(missing)} extra={sorted(extra)}")


def run_train(config: RunConfig, *, force: bool = False) -> list[StageResult]:
    _validate_order()
    ctx = build_context(config)
    return run_plan(ctx, train_plan(), force=force)


def run_infer(config: RunConfig, *, force: bool = False) -> list[StageResult]:
    _validate_order()
    ctx = build_context(config)
    return run_plan(ctx, infer_plan(), force=force)
