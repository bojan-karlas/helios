"""``helios fit <noun>`` -- train models (one bundle per CV fold).

Each ``fit`` command pools one or more ``--dataset`` training cohorts and writes
per-fold model bundles. There is no run-all ``fit``: training order (mil →
concept/morphology/risk sub-scores → risk helios) is sequenced by deployment
scripts, since stages may bind to different cohorts.
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.cli._common import (
    LOG_DIR,
    MAX_CPUS,
    MODELS_ROOT,
    PREPROCESSING_ROOT,
    RAY_ADDRESS,
    REMOVED_FEATURES_ROOT,
    REMOVED_OUTPUT_ROOT,
    RESULTS_ROOT,
    RUN_TAG,
    SPLITS_FILE,
    console_progress,
    csv,
    datasets_arg,
)

fit_app = typer.Typer(add_completion=False, help="Train models (per-fold bundles).")

_DATASET = typer.Option(..., "--dataset", help="Training cohort root(s) or cohort file(s); pooled across folds.")
_CONFIG = typer.Option(None, "--config", "-c", help="Override config YAML.")
_FORCE = typer.Option(False, "--force", help="Re-fit even if a model bundle exists.")
_FILTER = typer.Option(None, "--filter", help="pandas-query over image_metadata to subset training rows.")


def _reject_augmentation_target(value: str | None) -> None:
    if value is not None:
        raise typer.BadParameter(
            "--augmentation-target has been removed. Use --augmentation PATH to select a prepared directory."
        )


@fit_app.command("mil")
def mil(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    size_mm: float | None = typer.Option(None, "--size-mm", help="Tile resolution to train on."),
    model: str | None = typer.Option(None, "--model", help="Foundation model providing tile features."),
    target: str | None = typer.Option(None, "--target", help="Outcome column to train against."),
    augmentation: list[str] | None = typer.Option(
        None, "--augmentation", help="Prepared augmentation directory; repeatable. Replaces YAML selections."
    ),
    augmentation_target: str | None = typer.Option(
        None, "--augmentation-target", hidden=True, callback=_reject_augmentation_target,
    ),
    no_augmentations: bool = typer.Option(False, "--no-augmentations", help="Disable YAML augmentation selections."),
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
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.mil import fit_mil

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots, require_models=True)
    if no_augmentations and augmentation:
        raise typer.BadParameter("--augmentation and --no-augmentations are mutually exclusive.")
    overrides = {
        "size_mm": size_mm,
        "model": model,
        "target": target,
        "augmentations": [] if no_augmentations else augmentation,
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
        datasets=datasets, roots=roots, parallel=parallel, force=force,
        progress=console_progress(
            results_root or models_root or datasets[0].root, log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("fit mil: done")


@fit_app.command("concept")
def concept(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    n_bootstrap: int | None = typer.Option(None, "--n-bootstrap", help="Bootstrap replicas per concept."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Train per-concept probes (+ concept risk + residual correction)."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.concept import fit_concept

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots, require_models=True)
    params = load_stage_params("fit", "concept", config_path=config, overrides={"n_bootstrap": n_bootstrap})
    fit_concept(
        datasets=datasets, roots=roots, parallel=parallel, force=force,
        progress=console_progress(
            results_root or models_root or datasets[0].root, log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("fit concept: done")


@fit_app.command("morphology")
def morphology(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
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
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.morphology import fit_morphology

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots, require_models=True)
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
        datasets=datasets, roots=roots, parallel=parallel, force=force,
        progress=console_progress(
            results_root or models_root or datasets[0].root, log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("fit morphology: done")


@fit_app.command("risk")
def risk(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
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
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.risk import fit_risk

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots, require_models=True)
    overrides = {"risk": csv(risk), "target": target, "estimator": estimator}
    params = load_stage_params("fit", "risk", config_path=config, overrides=overrides)
    fit_risk(
        datasets=datasets, roots=roots, parallel=parallel, force=force,
        progress=console_progress(
            results_root or models_root or datasets[0].root, log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("fit risk: done")
