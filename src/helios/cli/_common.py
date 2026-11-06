"""Shared helpers for the verb sub-apps (``prep``, ``fit``, ``predict``, ``report``).

Each verb is a Typer sub-app whose commands are the nouns. A command is a thin
wrapper: it resolves effective parameters (defaults + ``--config`` + flags),
builds a console Progress backed by a ProgressReporter, and calls the importable
stage function. Comma-separated list flags REPLACE the configured list
(consistent with the config precedence rules).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from helios.data.cohort import ResolvedDataset
from helios.data.resolver import StorageDirs
from helios.progress import Progress, configure_logging, create_run_logs_dir
from helios.progress.reporter import ProgressReporter

if TYPE_CHECKING:
    from helios.stages._ray import ParallelConfig


@dataclass(frozen=True)
class Invocation:
    datasets: list[ResolvedDataset]
    dirs: StorageDirs
    parallel: ParallelConfig
    logs_dir: Path | None
    run_tag: str | None

    def progress(self) -> Progress:
        """Create logs only when the command is ready to call its stage."""
        from helios.stages._runtime import cohort_store

        progress = console_progress(logs_dir=self.logs_dir, run_tag=self.run_tag)
        for ds in self.datasets:
            store = cohort_store(ds, self.dirs)
            for role in ("source", "preprocessing", "models", "results"):
                location = getattr(store.resolver.dirs, f"{role}_dir")
                if role == "preprocessing" and location is None:
                    location = ds.root / "preprocessing"
                if location is not None:
                    progress.log(f"input {role} dataset={ds.logical_name}: {location}")
        if self.logs_dir is None:
            progress.log("Logs will NOT be saved; console feedback only.", level=logging.WARNING)
        return progress


def prepare_invocation(
    dataset: list[str],
    *,
    preprocessing_dir: Path | None = None,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
    splits_file: Path | None = None,
    ray_address: str | None = None,
    max_cpus: int | None = None,
    where: str | None = None,
    logs_dir: Path | None = None,
    run_tag: str | None = None,
    output_dir: Path | None = None,
) -> Invocation:
    """Resolve shared command inputs without hiding the explicit Typer options."""
    from helios.stages._runtime import ParallelConfig, resolve_datasets

    dirs = StorageDirs(
        preprocessing_dir=preprocessing_dir, models_dir=models_dir,
        results_dir=results_dir, splits_file=splits_file, output_dir=output_dir,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=where, dirs=dirs)
    if logs_dir is None and output_dir is not None:
        logs_dir = output_dir / "logs"
    return Invocation(datasets, dirs, parallel, logs_dir, run_tag)


OUTPUT_DIR = typer.Option(None, "--output-dir", help="Save defaults: DIR/models, DIR/results, DIR/logs; not inputs.")
PREPROCESSING_DIR = typer.Option(None, "--preprocessing-dir", help="Read/write preprocessing directory.")
MODELS_DIR = typer.Option(None, "--models-dir", help="Read/write model directory; overrides output default.")
RESULTS_DIR = typer.Option(None, "--results-dir", help="Results base; artifacts are grouped by cohort name.")
SPLITS_FILE = typer.Option(None, "--splits-file", help="Explicit cross-validation splits CSV path: image_id, fold, split, or image_id and test fold.")
LOGS_DIR = typer.Option(None, "--logs-dir", help="Base directory for run logs (<logs_dir>/<run_id>/).")
RUN_TAG = typer.Option(None, "--run-tag", help="Optional tag appended to <run_id> in ASCII kebab-case.")
RAY_ADDRESS = typer.Option(None, "--ray-address", help="Attach to an existing Ray cluster (enables parallelism).")
MAX_CPUS = typer.Option(
    None, "--max-cpus", help="CPU budget; values above 1 start a local Ray runtime (default: serial, single process)."
)


def csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated flag into a list (``None`` stays ``None``)."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_floats(value: str | None) -> list[float] | None:
    items = csv(value)
    return [float(x) for x in items] if items is not None else None


def datasets_arg(values: list[str] | None) -> list[str]:
    """Flatten ``--dataset`` (repeatable, each possibly comma-separated) to a list."""
    out: list[str] = []
    for value in values or []:
        out.extend(item.strip() for item in value.split(",") if item.strip())
    if not out:
        raise ValueError("at least one --dataset is required")
    return out


def console_progress(
    *,
    logs_dir: str | Path | None = None,
    run_tag: str | None = None,
) -> Progress:
    """A console Progress that writes status.json / events.jsonl / run.log under <logs_dir>/<run_id>/."""
    if logs_dir is None:
        configure_logging()
        return Progress.console()
    run_id, run_dir = create_run_logs_dir(logs_dir, run_tag)
    logger = configure_logging(run_dir)
    reporter = ProgressReporter(run_dir)
    reporter.update(run_id=run_id, logs_dir=str(run_dir))
    logger.info(f"run_id={run_id} logs_dir={run_dir}")
    return Progress.console(reporter=reporter)
