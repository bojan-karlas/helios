"""``helios fit <noun>`` — train models (one bundle per CV fold).

Each ``fit`` command pools one or more ``--dataset`` training cohorts and writes
per-fold model bundles. There is no run-all ``fit``: training order (mil →
concept/morphology/risk sub-scores → risk helios) is sequenced by deployment
scripts, since stages may bind to different cohorts.
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.cli._common import console_progress, csv, datasets_arg
from helios.config.stage_config import load_stage_params
from helios.data.cohort import resolve_datasets
from helios.stages.concept import fit_concept
from helios.stages.mil import fit_mil
from helios.stages.morphology import fit_morphology
from helios.stages.risk import fit_risk

fit_app = typer.Typer(add_completion=False, help="Train models (per-fold bundles).")

_DATASET = typer.Option(..., "--dataset", help="Training cohort root(s) or cohort file(s); pooled across folds.")
_OUTPUT_ROOT = typer.Option(None, "--output-root", help="Write model bundles here instead of the first cohort.")
_CONFIG = typer.Option(None, "--config", "-c", help="Override config YAML.")
_FORCE = typer.Option(False, "--force", help="Re-fit even if a model bundle exists.")
_FILTER = typer.Option(
    None, "--filter", help="pandas-query over image_metadata to subset training rows."
)


@fit_app.command("mil")
def mil(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    size_mm: float | None = typer.Option(None, "--size-mm", help="Tile resolution to train on."),
    model: str | None = typer.Option(None, "--model", help="Foundation model providing tile features."),
    target: str | None = typer.Option(None, "--target", help="Outcome column to train against."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train the whole-slide A-MIL model."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {"size_mm": size_mm, "model": model, "target": target, "device": device}
    params = load_stage_params("fit", "mil", config_path=config, overrides=overrides)
    fit_mil(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("fit mil: done")


@fit_app.command("concept")
def concept(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    n_bootstrap: int | None = typer.Option(None, "--n-bootstrap", help="Bootstrap replicas per concept."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train per-concept probes (+ concept risk + residual correction)."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    params = load_stage_params("fit", "concept", config_path=config, overrides={"n_bootstrap": n_bootstrap})
    fit_concept(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("fit concept: done")


@fit_app.command("morphology")
def morphology(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    attention_percentile: float | None = typer.Option(
        None, "--attention-percentile", help="Attention filter percentile."
    ),
    n_pca: int | None = typer.Option(None, "--n-pca", help="PCA components before clustering."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train the patch-morphology clustering risk model."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {"attention_percentile": attention_percentile, "n_pca": n_pca}
    params = load_stage_params("fit", "morphology", config_path=config, overrides=overrides)
    fit_morphology(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("fit morphology: done")


@fit_app.command("risk")
def risk(
    dataset: list[str] = _DATASET,
    output_root: Path | None = _OUTPUT_ROOT,
    risk: str | None = typer.Option(
        None, "--risk", help="Comma list of risk kinds (cellular,staging,clinical,helios)."
    ),
    target: str | None = typer.Option(None, "--target", help="Outcome column to train against."),
    estimator: str | None = typer.Option(None, "--estimator", help="Simple estimator to fit."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train the simple risk sub-models (and HELIOS fusion)."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {"risk": csv(risk), "target": target, "estimator": estimator}
    params = load_stage_params("fit", "risk", config_path=config, overrides=overrides)
    fit_risk(
        datasets=datasets, output_root=output_root, force=force,
        progress=console_progress(output_root or datasets[0].root), **params,
    )
    typer.echo("fit risk: done")
