"""``helios predict <noun>`` — run inference (per-cohort, K folds reduced).

Each ``predict`` command scores one or more ``--dataset`` cohorts with the fitted
models; the per-fold scores are reduced (out-of-fold on a CV cohort /
ensemble-average on deploy) inside each stage. There is no run-all ``predict``:
ordering (mil → concept/morphology/risk sub-scores → risk helios) is sequenced by
deployment scripts.
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.cli._common import console_progress, csv, datasets_arg
from helios.config.stage_config import load_stage_params
from helios.data.cohort import resolve_datasets
from helios.stages.concept import predict_concept
from helios.stages.mil import predict_mil
from helios.stages.morphology import predict_morphology
from helios.stages.risk import predict_risk

predict_app = typer.Typer(add_completion=False, help="Run inference (per-cohort).")

_DATASET = typer.Option(
    ..., "--dataset", help="Cohort root(s) or cohort file(s) to score; repeatable or comma-separated."
)
_OUTPUT_ROOT = typer.Option(
    None, "--output-root", help="Write predictions here (source stays read-only); single --dataset only."
)
_CONFIG = typer.Option(None, "--config", "-c", help="Override config YAML.")
_FORCE = typer.Option(False, "--force", help="Re-score even if outputs exist.")
_FILTER = typer.Option(
    None, "--filter", help="pandas-query over image_metadata to subset scored images."
)


@predict_app.command("mil")
def mil(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    size_mm: float | None = typer.Option(None, "--size-mm", help="Tile resolution to score on."),
    model: str | None = typer.Option(None, "--model", help="Foundation model providing tile features."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Score whole-slide A-MIL and reduce whole-image risk."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {"size_mm": size_mm, "model": model, "device": device}
    params = load_stage_params("predict", "mil", config_path=config, overrides=overrides)
    predict_mil(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("predict mil: done")


@predict_app.command("concept")
def concept(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Predict concepts and the concept-based risk score."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    params = load_stage_params("predict", "concept", config_path=config)
    predict_concept(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("predict concept: done")


@predict_app.command("morphology")
def morphology(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Assign clusters, score cluster presence, and produce morphology risk."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    params = load_stage_params("predict", "morphology", config_path=config)
    predict_morphology(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("predict morphology: done")


@predict_app.command("risk")
def risk(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    risk: str | None = typer.Option(
        None, "--risk", help="Comma list of risk kinds (cellular,staging,clinical,helios)."
    ),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Score the simple risk sub-models (and HELIOS fusion + survival)."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    params = load_stage_params("predict", "risk", config_path=config, overrides={"risk": csv(risk)})
    predict_risk(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("predict risk: done")
