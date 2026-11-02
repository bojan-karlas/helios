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
    LOGS_DIR,
    MAX_CPUS,
    MODELS_DIR,
    OUTPUT_DIR,
    PREPROCESSING_DIR,
    RAY_ADDRESS,
    RESULTS_DIR,
    RUN_TAG,
    SPLITS_FILE,
    csv,
    prepare_invocation,
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
    preprocessing_dir: Path | None = PREPROCESSING_DIR,
    models_dir: Path | None = MODELS_DIR,
    results_dir: Path | None = RESULTS_DIR,
    splits_file: Path | None = SPLITS_FILE,
    logs_dir: Path | None = LOGS_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_dir: Path | None = OUTPUT_DIR,
    size_mm: float | None = typer.Option(None, "--size-mm", help="Tile resolution to score on."),
    model: str | None = typer.Option(None, "--model", help="Foundation model providing tile features."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Score whole-slide A-MIL and reduce whole-image risk."""
    from helios.config.stage_config import load_stage_params
    from helios.stages.mil import predict_mil

    invocation = prepare_invocation(
        dataset,
        preprocessing_dir=preprocessing_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        splits_file=splits_file,
        ray_address=ray_address,
        max_cpus=max_cpus,
        where=filter,
        logs_dir=logs_dir,
        run_tag=run_tag, output_dir=output_dir,
    )
    overrides = {"size_mm": size_mm, "model": model, "device": device}
    params = load_stage_params("predict", "mil", config_path=config, overrides=overrides)
    predict_mil(
        datasets=invocation.datasets, dirs=invocation.dirs, parallel=invocation.parallel, overwrite=overwrite,
        progress=invocation.progress(),
        **params,
    )
    typer.echo("predict mil: done")


@predict_app.command("concept")
def concept(
    dataset: list[str] = _DATASET,
    preprocessing_dir: Path | None = PREPROCESSING_DIR,
    models_dir: Path | None = MODELS_DIR,
    results_dir: Path | None = RESULTS_DIR,
    splits_file: Path | None = SPLITS_FILE,
    logs_dir: Path | None = LOGS_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_dir: Path | None = OUTPUT_DIR,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Predict concepts and the concept-based risk score."""
    from helios.config.stage_config import load_stage_params
    from helios.stages.concept import predict_concept

    invocation = prepare_invocation(
        dataset,
        preprocessing_dir=preprocessing_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        splits_file=splits_file,
        ray_address=ray_address,
        max_cpus=max_cpus,
        where=filter,
        logs_dir=logs_dir,
        run_tag=run_tag, output_dir=output_dir,
    )
    params = load_stage_params("predict", "concept", config_path=config)
    predict_concept(
        datasets=invocation.datasets, dirs=invocation.dirs, parallel=invocation.parallel, overwrite=overwrite,
        progress=invocation.progress(),
        **params,
    )
    typer.echo("predict concept: done")


@predict_app.command("morphology")
def morphology(
    dataset: list[str] = _DATASET,
    preprocessing_dir: Path | None = PREPROCESSING_DIR,
    models_dir: Path | None = MODELS_DIR,
    results_dir: Path | None = RESULTS_DIR,
    splits_file: Path | None = SPLITS_FILE,
    logs_dir: Path | None = LOGS_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_dir: Path | None = OUTPUT_DIR,
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Assign clusters, score cluster presence, and produce morphology risk."""
    from helios.config.stage_config import load_stage_params
    from helios.stages.morphology import predict_morphology

    invocation = prepare_invocation(
        dataset,
        preprocessing_dir=preprocessing_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        splits_file=splits_file,
        ray_address=ray_address,
        max_cpus=max_cpus,
        where=filter,
        logs_dir=logs_dir,
        run_tag=run_tag, output_dir=output_dir,
    )
    params = load_stage_params("predict", "morphology", config_path=config)
    predict_morphology(
        datasets=invocation.datasets, dirs=invocation.dirs, parallel=invocation.parallel, overwrite=overwrite,
        progress=invocation.progress(),
        **params,
    )
    typer.echo("predict morphology: done")


@predict_app.command("risk")
def risk(
    dataset: list[str] = _DATASET,
    preprocessing_dir: Path | None = PREPROCESSING_DIR,
    models_dir: Path | None = MODELS_DIR,
    results_dir: Path | None = RESULTS_DIR,
    splits_file: Path | None = SPLITS_FILE,
    logs_dir: Path | None = LOGS_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_dir: Path | None = OUTPUT_DIR,
    risk: str | None = typer.Option(
        None, "--risk", help="Comma list of risk kinds (cellular,staging,clinical,helios)."
    ),
    filter: str | None = _FILTER,
    config: Path | None = _CONFIG,
    overwrite: bool = _OVERWRITE,
) -> None:
    """Score the simple risk sub-models (and HELIOS fusion + survival)."""
    from helios.config.stage_config import load_stage_params
    from helios.stages.risk import predict_risk

    invocation = prepare_invocation(
        dataset,
        preprocessing_dir=preprocessing_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        splits_file=splits_file,
        ray_address=ray_address,
        max_cpus=max_cpus,
        where=filter,
        logs_dir=logs_dir,
        run_tag=run_tag, output_dir=output_dir,
    )
    params = load_stage_params("predict", "risk", config_path=config, overrides={"risk": csv(risk)})
    predict_risk(
        datasets=invocation.datasets, dirs=invocation.dirs, parallel=invocation.parallel, overwrite=overwrite,
        progress=invocation.progress(),
        **params,
    )
    typer.echo("predict risk: done")
