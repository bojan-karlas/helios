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
    aug_swap_prob: float | None = typer.Option(
        None, "--aug-swap-prob", help="Per-tile probability of swapping in the stain-augmented embedding."
    ),
    aug_mode: str | None = typer.Option(
        None, "--aug-mode", help="'replace' swaps a tile's embedding in place; 'augment' appends it to the bag."
    ),
    n_branches: int | None = typer.Option(None, "--n-branches", help="Number of gated-attention branches (K)."),
    num_heads: int | None = typer.Option(None, "--num-heads", help="Attention heads per branch."),
    dropout_rate: float | None = typer.Option(None, "--dropout-rate", help="Dropout rate throughout the network."),
    lambda_div: float | None = typer.Option(
        None, "--lambda-div", help="Weight of the branch-attention diversity loss."
    ),
    lr: float | None = typer.Option(None, "--lr", help="Adam learning rate."),
    weight_decay: float | None = typer.Option(None, "--weight-decay", help="Adam weight decay."),
    max_epochs: int | None = typer.Option(None, "--max-epochs", help="Max training epochs per fold."),
    patience: int | None = typer.Option(None, "--patience", help="Early-stopping patience (epochs)."),
    val_frac: float | None = typer.Option(
        None, "--val-frac", help="Fraction of each fold's train slides held out for early-stopping."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train the whole-slide A-MIL model."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {
        "size_mm": size_mm,
        "model": model,
        "target": target,
        "aug_swap_prob": aug_swap_prob,
        "aug_mode": aug_mode,
        "n_branches": n_branches,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "lambda_div": lambda_div,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "val_frac": val_frac,
        "seed": seed,
        "device": device,
    }
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
    target: str | None = typer.Option(None, "--target", help="Outcome column to train against."),
    attention_cutoff: float | None = typer.Option(
        None, "--attention-cutoff", help="Cumulative attention-mass cutoff for the tile filter."
    ),
    n_pca: int | None = typer.Option(None, "--n-pca", help="PCA components before clustering."),
    k_micro: int | None = typer.Option(None, "--k-micro", help="FAISS k-means over-cluster count."),
    k_neighbors: int | None = typer.Option(None, "--k-neighbors", help="kNN graph neighbors for Leiden."),
    resolution: float | None = typer.Option(None, "--resolution", help="Leiden resolution parameter."),
    significance_alpha: float | None = typer.Option(
        None, "--significance-alpha", help="BH-adjusted p-value threshold for cluster exclusion."
    ),
    estimator: str | None = typer.Option(None, "--estimator", help="Cluster-presence risk estimator."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train the patch-morphology clustering risk model."""
    datasets = resolve_datasets(datasets_arg(dataset), where=filter)
    overrides = {
        "target": target,
        "attention_cutoff": attention_cutoff,
        "n_pca": n_pca,
        "k_micro": k_micro,
        "k_neighbors": k_neighbors,
        "resolution": resolution,
        "significance_alpha": significance_alpha,
        "estimator": estimator,
        "seed": seed,
    }
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
