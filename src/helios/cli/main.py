"""HELIOS command-line interface.

``helios`` exposes four verbs — ``prep`` (derive data artifacts), ``fit`` (train
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

from helios.cli._common import console_progress, datasets_arg
from helios.cli.fit import fit_app
from helios.cli.predict import predict_app
from helios.cli.prep import prep_app
from helios.config.stage_config import load_stage_params
from helios.data.cohort import resolve_datasets
from helios.stages.report import build_report

app = typer.Typer(add_completion=False, help="HELIOS computational-pathology pipeline.")
app.add_typer(prep_app, name="prep")
app.add_typer(fit_app, name="fit")
app.add_typer(predict_app, name="predict")


@app.command("report")
def report(
    dataset: list[str] = typer.Option(
        ..., "--dataset", help="Cohort root(s) or cohort file(s) to report; repeatable or comma-separated."
    ),
    output_root: Path | None = typer.Option(
        None, "--output-root", help="Write reports here (source stays read-only); single --dataset only."
    ),
    no_assets: bool = typer.Option(False, "--no-assets", help="Skip rendering PNG assets."),
    no_summarize: bool = typer.Option(False, "--no-summarize", help="Skip the cohort summary join."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML."),
    force: bool = typer.Option(False, "--force", help="Re-render even if reports exist."),
) -> None:
    """Build the per-case reports and cohort summary (web-app data package)."""
    datasets = resolve_datasets(datasets_arg(dataset))
    overrides = {
        "assets": False if no_assets else None,
        "summarize": False if no_summarize else None,
    }
    params = load_stage_params("report", "build", config_path=config, overrides=overrides)
    build_report(
        datasets=datasets,
        output_root=output_root,
        force=force,
        progress=console_progress(output_root or datasets[0].root),
        **params,
    )
    typer.echo("report: done")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
