"""HELIOS command-line interface.

``helios`` exposes four verbs -- ``prep`` (derive data artifacts), ``fit`` (train
models), ``predict`` (run inference), and ``report`` (build the output package).
``prep``/``fit``/``predict`` are Typer sub-apps whose commands are *nouns* that map
1:1 to importable stage functions under :mod:`helios.stages`; ``report`` is a
single command (no nouns). The wrappers only resolve config and wire Progress.
Run-everything sequencing lives in bash scripts, not in the CLI (except bare
``helios prep``, which runs all prep nouns for a fresh cohort).
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
    console_progress,
    datasets_arg,
    resolved_results_root,
)
from helios.cli.fit import fit_app
from helios.cli.predict import predict_app
from helios.cli.prep import prep_app

app = typer.Typer(add_completion=False, help="HELIOS computational-pathology pipeline.")
app.add_typer(prep_app, name="prep")
app.add_typer(fit_app, name="fit")
app.add_typer(predict_app, name="predict")
list_app = typer.Typer(help="Discover prepared artifacts.", add_completion=False)
app.add_typer(list_app, name="list")


@list_app.command("augmentations")
def list_augmentations(
    dataset: list[str] | None = typer.Option(None, "--dataset", help="Dataset roots or cohort YAML files."),
    root: Path | None = typer.Option(None, "--root", help="Search an explicit augmentation storage directory."),
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
) -> None:
    """List saved definitions and available files, not a guarantee of completeness."""
    import json

    from helios.data.augmentation import Augmentation, discover_augmentations
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import cohort_store, resolve_datasets

    if not dataset and root is None:
        raise typer.BadParameter("Supply --dataset or --root.")
    found: dict[Path, Augmentation] = {}
    if root is not None:
        found.update((a.root, a) for a in discover_augmentations(root))
    if dataset:
        roots = StorageRoots(preprocessing_root=preprocessing_root)
        for ds in resolve_datasets(datasets_arg(dataset), roots=roots):
            base = cohort_store(ds, roots).resolver.roots.for_role("preprocessing") / "augmentations"
            if base.is_dir():
                found.update((a.root, a) for a in discover_augmentations(base))
            for path in ds.augmentations:
                a = Augmentation.load(path)
                found[a.root] = a
    rows = []
    for a in found.values():
        features = {}
        for partition in sorted((a.root / "features").glob("size_mm=*/model=*")):
            features[str(partition.relative_to(a.root))] = len(list(partition.glob("*/*.hdf5")))
        rows.append({
            "path": str(a.root), "source_dataset_id": a.source_dataset_id,
            "method": a.method, "parameters": a.parameters,
            "checkpoint": str(a.checkpoint) if a.checkpoint is not None else None,
            "tile_files": len(list((a.root / "tiles").glob("size_mm=*/*/*.hdf5"))),
            "feature_files": features,
        })
    typer.echo(json.dumps(rows, indent=2))


@app.command("report")
def report(
    dataset: list[str] = typer.Option(
        ..., "--dataset", help="Cohort root(s) or cohort file(s) to report; repeatable or comma-separated."
    ),
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    removed_features_root: Path | None = REMOVED_FEATURES_ROOT,
    models_root: Path | None = MODELS_ROOT,
    results_root: Path | None = RESULTS_ROOT,
    log_dir: Path | None = LOG_DIR,
    run_tag: str | None = RUN_TAG,
    ray_address: str | None = RAY_ADDRESS,
    max_cpus: int | None = MAX_CPUS,
    output_root: Path | None = REMOVED_OUTPUT_ROOT,
    no_assets: bool = typer.Option(False, "--no-assets", help="Skip rendering PNG assets."),
    no_summarize: bool = typer.Option(False, "--no-summarize", help="Skip the cohort summary join."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-render even if reports exist."),
) -> None:
    """Build the per-case reports and cohort summary (web-app data package)."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.report import build_report

    roots = StorageRoots(
        preprocessing_root=preprocessing_root,
        models_root=models_root, results_root=results_root,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), roots=roots)
    overrides = {
        "assets": False if no_assets else None,
        "summarize": False if no_summarize else None,
    }
    params = load_stage_params("report", "build", config_path=config, overrides=overrides)
    build_report(
        datasets=datasets,
        roots=roots,
        parallel=parallel,
        overwrite=overwrite,
        progress=console_progress(
            resolved_results_root(datasets[0], roots),
            log_dir=log_dir,
            run_tag=run_tag,
        ),
        **params,
    )
    typer.echo("report: done")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
