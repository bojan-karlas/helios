"""``helios prep <noun>`` -- derive data artifacts (no model fitting).

Bare ``helios prep`` runs every prep noun in order (splits → thumbnails → tiles →
augment → tile-features → cells → cell-features) using config/default parameters;
a noun runs just that stage with its own flags. Each command resolves effective
parameters (defaults + ``--config`` + flags), builds a console Progress, and
calls the importable stage function.
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.cli._common import (
    FEATURES_ROOT,
    MODELS_ROOT,
    PREPROCESSING_ROOT,
    REMOVED_OUTPUT_ROOT,
    RESULTS_ROOT,
    console_progress,
    csv,
    csv_floats,
    datasets_arg,
)

prep_app = typer.Typer(add_completion=False, help="Derive data artifacts (no fitting).")

_DATASET = typer.Option(..., "--dataset", help="Cohort root(s) or cohort file(s); repeatable or comma-separated.")
_DATASET_GROUP = typer.Option(
    None, "--dataset", help="Cohort root(s) or cohort file(s); repeatable or comma-separated. Required for bare `helios prep`."
)
_CONFIG = typer.Option(None, "--config", "-c", help="Override config YAML.")
_FORCE = typer.Option(False, "--force", help="Re-derive even if outputs exist.")


@prep_app.callback(invoke_without_command=True)
def prep_all(
    ctx: typer.Context,
    dataset: list[str] = _DATASET_GROUP,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Run every prep stage in order (deployment-style full prep for a cohort)."""
    if ctx.invoked_subcommand is not None:
        return
    if not dataset:
        raise typer.BadParameter("at least one --dataset is required", param_hint="--dataset")
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages import STAGES
    from helios.stages._runtime import resolve_datasets

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    progress = console_progress(results_root or models_root or datasets[0].root)
    for noun, stage in STAGES["prep"].items():
        params = load_stage_params("prep", noun, config_path=config)
        stage(datasets=datasets, roots=roots, force=force, progress=progress, **params)
    typer.echo("prep: done")


@prep_app.command("splits")
def splits(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    n_folds: int | None = typer.Option(None, "--n-folds", help="Number of CV folds."),
    val_fraction: float | None = typer.Option(
        None, "--val-fraction", help="Fraction of each fold's train rows carved off as val."
    ),
    stratify_by: str | None = typer.Option(
        None, "--stratify-by", help="Comma list of columns to stratify on (joint key if >1)."
    ),
    group_by: str | None = typer.Option(
        None,
        "--group-by",
        help="Column kept within one fold (e.g. patient_id). To disable grouping (no CLI"
        " override for 'None' -- flags can't distinguish unset from explicit-None), use a"
        " --config YAML with 'group_by: null' instead.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="RNG seed."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Generate cross-validation splits for a training cohort."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.splits import prep_splits

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {
        "n_folds": n_folds,
        "val_fraction": val_fraction,
        "stratify_by": csv(stratify_by),
        "group_by": group_by,
        "seed": seed,
    }
    params = load_stage_params("prep", "splits", config_path=config, overrides=overrides)
    prep_splits(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep splits: done")


@prep_app.command("thumbnails")
def thumbnails(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    max_size: int | None = typer.Option(None, "--max-size", help="Thumbnail longest-edge pixels."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Extract a low-resolution thumbnail per slide."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.thumbnails import prep_thumbnails

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    params = load_stage_params("prep", "thumbnails", config_path=config, overrides={"max_size": max_size})
    prep_thumbnails(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep thumbnails: done")


@prep_app.command("tiles")
def tiles(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    size_mm: str | None = typer.Option(None, "--size-mm", help="Comma list of resolutions."),
    tile_px: int | None = typer.Option(None, "--tile-px", help="Tile edge length in pixels."),
    stride_fraction: float | None = typer.Option(
        None, "--stride-fraction", help="Grid stride as a fraction of tile width (0.5 gives 50% overlap)."
    ),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Tile each slide and flag background tiles."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.tiles import prep_tiles

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {
        "size_mm": csv_floats(size_mm),
        "tile_px": tile_px,
        "stride_fraction": stride_fraction,
    }
    params = load_stage_params("prep", "tiles", config_path=config, overrides=overrides)
    prep_tiles(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep tiles: done")


@prep_app.command("augment")
def augment(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    target: str | None = typer.Option(None, "--target", help="Comma list of stain targets."),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", help="Comma list of user-trained MultiStain-CycleGAN G_A checkpoints."
    ),
    stain_model_root: Path | None = typer.Option(
        None, "--stain-model-root", help="Root for <target>/latest_net_G_A.pth checkpoint lookup."
    ),
    size_mm: str | None = typer.Option(None, "--size-mm", help="Comma list of resolutions."),
    model: str | None = typer.Option(None, "--model", help="Comma list of foundation models."),
    augmentation_batch_size: int | None = typer.Option(
        None, "--augmentation-batch-size", help="Tiles per CycleGAN forward pass."
    ),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Stain-augment tiles and extract augmented features."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.augment import prep_augment

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {
        "target": csv(target),
        "checkpoint": csv(checkpoint),
        "stain_model_root": stain_model_root,
        "size_mm": csv_floats(size_mm),
        "model": csv(model),
        "augmentation_batch_size": augmentation_batch_size,
        "device": device,
    }
    params = load_stage_params("prep", "augment", config_path=config, overrides=overrides)
    prep_augment(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep augment: done")


@prep_app.command("tile-features")
def tile_features(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    size_mm: str | None = typer.Option(None, "--size-mm", help="Comma list of resolutions, e.g. 0.25,0.5."),
    model: str | None = typer.Option(None, "--model", help="Comma list of foundation models."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Tiles per forward pass."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Extract foundation tile features for staged tiles."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.tile_features import prep_tile_features

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {"size_mm": csv_floats(size_mm), "model": csv(model), "batch_size": batch_size, "device": device}
    params = load_stage_params("prep", "tile-features", config_path=config, overrides=overrides)
    prep_tile_features(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep tile-features: done")


@prep_app.command("cells")
def cells(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    patch_px: int | None = typer.Option(None, "--patch-px", help="Tumor patch edge length in pixels."),
    device: str | None = typer.Option(None, "--device", help="Torch device (cuda, cpu)."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Segment cells, cut tumor patches, and classify mitoses."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.cells import prep_cells

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {"patch_px": patch_px, "device": device}
    params = load_stage_params("prep", "cells", config_path=config, overrides=overrides)
    prep_cells(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep cells: done")


@prep_app.command("cell-features")
def cell_features(
    dataset: list[str] = _DATASET,
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    size_mm: str | None = typer.Option(None, "--size-mm", help="Comma list of resolutions."),
    config: Path | None = _CONFIG,
    force: bool = _FORCE,
) -> None:
    """Aggregate per-tile cell counts and compute cellular features."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import resolve_datasets
    from helios.stages.cell_features import prep_cell_features

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
        models_root=models_root, results_root=results_root,
    )
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    params = load_stage_params("prep", "cell-features", config_path=config, overrides={"size_mm": csv_floats(size_mm)})
    prep_cell_features(
        datasets=datasets, roots=roots, force=force,
        progress=console_progress(results_root or models_root or datasets[0].root), **params,
    )
    typer.echo("prep cell-features: done")
