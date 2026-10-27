"""``helios predict <noun>`` -- run inference (per-cohort, K folds reduced).

Each ``predict`` command scores one or more ``--dataset`` cohorts with the fitted
models; the per-fold scores are reduced (out-of-fold on a CV cohort /
ensemble-average on deploy) inside each stage. There is no run-all ``predict``:
ordering (mil → concept/morphology/risk sub-scores → risk helios) is sequenced by
deployment scripts.
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
    resolved_results_root,
)

predict_app = typer.Typer(add_completion=False, help="Run inference (per-cohort).")

_DATASET = typer.Option(
    ..., "--dataset", help="Cohort root(s) or cohort file(s) to score; repeatable or comma-separated."
)
_CONFIG = typer.Option(None, "--config", "-c", help="Override config YAML.")
_OVERWRITE = typer.Option(False, "--overwrite", help="Re-score even if outputs exist.")
_FILTER = typer.Option(
    None, "--filter", help="pandas-query over image_metadata to subset scored images."
)


@predict_app.command("mil")
def mil(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    removed_features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    size_mm: float | None = typer.Option(None, "--size-mm", help="Tile resolution to score on."),
    model: str | None = typer.Option(None, "--model", help="Foundation model providing tile features."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Score whole-slide A-MIL and reduce whole-image risk."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.mil import predict_mil

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots)
    overrides = {"size_mm": size_mm, "model": model, "device": device}
    params = load_stage_params("predict", "mil", config_path=config, overrides=overrides)
    predict_mil(
        datasets=datasets, roots=roots, parallel=parallel, overwrite=overwrite,
        progress=console_progress(
            resolved_results_root(datasets[0], roots), log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("predict mil: done")


@predict_app.command("concept")
def concept(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    removed_features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Predict concepts and the concept-based risk score."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.concept import predict_concept

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots)
    params = load_stage_params("predict", "concept", config_path=config)
    predict_concept(
        datasets=datasets, roots=roots, parallel=parallel, overwrite=overwrite,
        progress=console_progress(
            resolved_results_root(datasets[0], roots), log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("predict concept: done")


@predict_app.command("morphology")
def morphology(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    removed_features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    splits_file: Path | None = SPLITS_FILE,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Assign clusters, score cluster presence, and produce morphology risk."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.morphology import predict_morphology

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots)
    params = load_stage_params("predict", "morphology", config_path=config)
    predict_morphology(
        datasets=datasets, roots=roots, parallel=parallel, overwrite=overwrite,
        progress=console_progress(
            resolved_results_root(datasets[0], roots), log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("predict morphology: done")


@predict_app.command("risk")
def risk(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    removed_features_root: Path | None = REMOVED_FEATURES_ROOT,
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
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Score the simple risk sub-models (and HELIOS fusion + survival)."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.risk import predict_risk

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
        splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=filter, roots=roots)
    params = load_stage_params("predict", "risk", config_path=config, overrides={"risk": csv(risk)})
    predict_risk(
        datasets=datasets, roots=roots, parallel=parallel, overwrite=overwrite,
        progress=console_progress(
            resolved_results_root(datasets[0], roots), log_dir=log_dir, run_tag=run_tag
        ),
        **params,
    )
    typer.echo("predict risk: done")
