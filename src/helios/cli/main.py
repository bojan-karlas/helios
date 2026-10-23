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
    FEATURES_ROOT,
    LOG_DIR,
    MAX_CPUS,
    MODELS_ROOT,
    PREPROCESSING_ROOT,
    RAY_ADDRESS,
    REMOVED_OUTPUT_ROOT,
    RESULTS_ROOT,
    RUN_TAG,
    console_progress,
    datasets_arg,
)
from helios.cli.fit import fit_app
from helios.cli.predict import predict_app
from helios.cli.prep import prep_app

app = typer.Typer(add_completion=False, help="HELIOS computational-pathology pipeline.")
app.add_typer(prep_app, name="prep")
app.add_typer(fit_app, name="fit")
app.add_typer(predict_app, name="predict")


@app.command("report")
def report(
    dataset: list[str] = typer.Option(
        ..., "--dataset", help="Cohort root(s) or cohort file(s) to report; repeatable or comma-separated."
    ),
    preprocessing_root: Path | None = PREPROCESSING_ROOT,
    features_root: Path | None = FEATURES_ROOT,
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
    force: bool = typer.Option(False, "--force", help="Re-render even if reports exist."),
) -> None:
    """Build the per-case reports and cohort summary (web-app data package)."""
    from helios.config.stage_config import load_stage_params
    from helios.data.resolver import StorageRoots
    from helios.stages._runtime import ParallelConfig, resolve_datasets
    from helios.stages.report import build_report

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, features_root=features_root,
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
        force=force,
        progress=console_progress(
            results_root or models_root or datasets[0].root,
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
