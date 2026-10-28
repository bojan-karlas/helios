"""Shared helpers for the verb sub-apps (``prep``, ``fit``, ``predict``, ``report``).

Each verb is a Typer sub-app whose commands are the nouns. A command is a thin
wrapper: it resolves effective parameters (defaults + ``--config`` + flags),
builds a console Progress backed by a ProgressReporter, and calls the importable
stage function. Comma-separated list flags REPLACE the configured list
(consistent with the config precedence rules).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from helios.data.cohort import ResolvedDataset
from helios.data.resolver import StorageRoots
from helios.progress import Progress, configure_logging, create_run_log_dir
from helios.progress.reporter import ProgressReporter

if TYPE_CHECKING:
    from helios.stages._ray import ParallelConfig


@dataclass(frozen=True)
class Invocation:
    datasets: list[ResolvedDataset]
    roots: StorageRoots
    parallel: ParallelConfig
    log_dir: Path | None
    run_tag: str | None

    def progress(self) -> Progress:
        """Create logs only when the command is ready to call its stage."""
        return console_progress(
            resolved_results_root(self.datasets[0], self.roots),
            log_dir=self.log_dir,
            run_tag=self.run_tag,
        )


def prepare_invocation(
    dataset: list[str],
    *,
    preprocessing_root: Path | None = None,
    models_root: Path | None = None,
    results_root: Path | None = None,
    splits_file: Path | None = None,
    ray_address: str | None = None,
    max_cpus: int | None = None,
    where: str | None = None,
    require_models: bool = False,
    log_dir: Path | None = None,
    run_tag: str | None = None,
) -> Invocation:
    """Resolve shared command inputs without hiding the explicit Typer options."""
    from helios.stages._runtime import ParallelConfig, resolve_datasets

    roots = StorageRoots(
        preprocessing_root=preprocessing_root, models_root=models_root,
        results_root=results_root, splits_file=splits_file,
    )
    parallel = ParallelConfig(ray_address=ray_address, max_cpus=max_cpus)
    datasets = resolve_datasets(datasets_arg(dataset), where=where, roots=roots, require_models=require_models)
    return Invocation(datasets, roots, parallel, log_dir, run_tag)


def reject_output_root(value: Path | None) -> None:
    if value is not None:
        raise typer.BadParameter(
            "--output-root has been removed. Use --preprocessing-root, --models-root, "
            "and/or --results-root to select artifact destinations."
        )


def reject_features_root(value: Path | None) -> None:
    if value is not None:
        raise typer.BadParameter(
            "--features-root has been removed. Feature artifacts are stored under the preprocessing root; "
            "use --preprocessing-root to select their destination."
        )


PREPROCESSING_ROOT = typer.Option(None, "--preprocessing-root", help="Preprocessing base; single dataset only.")
MODELS_ROOT = typer.Option(None, "--models-root", help="Shared model base; required for pooled training.")
RESULTS_ROOT = typer.Option(None, "--results-root", help="Results base; artifacts are grouped by cohort name.")
SPLITS_FILE = typer.Option(None, "--splits-file", help="Explicit cross-validation splits CSV path.")
LOG_DIR = typer.Option(None, "--log-dir", help="Base directory for run logs (<log_dir>/<run_id>/).")
RUN_TAG = typer.Option(None, "--run-tag", help="Optional tag appended to <run_id> in ASCII kebab-case.")
RAY_ADDRESS = typer.Option(None, "--ray-address", help="Attach to an existing Ray cluster (enables parallelism).")
MAX_CPUS = typer.Option(
    None, "--max-cpus", help="CPU budget; values above 1 start a local Ray runtime (default: serial, single process)."
)
REMOVED_OUTPUT_ROOT = typer.Option(None, "--output-root", hidden=True, callback=reject_output_root)
REMOVED_FEATURES_ROOT = typer.Option(None, "--features-root", hidden=True, callback=reject_features_root)


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


def resolved_results_root(dataset: ResolvedDataset, roots: StorageRoots) -> Path:
    """Use the same results destination as artifact resolution."""
    from helios.stages._runtime import cohort_store

    return cohort_store(dataset, roots).resolver.roots.for_role("results")


def console_progress(
    output_root: str | Path,
    *,
    log_dir: str | Path | None = None,
    run_tag: str | None = None,
) -> Progress:
    """A console Progress that writes status.json / events.jsonl / run.log under <log_dir>/<run_id>/."""
    base_logs = Path(log_dir) if log_dir is not None else Path(output_root) / "logs"
    run_id, run_dir = create_run_log_dir(base_logs, run_tag)
    logger = configure_logging(run_dir)
    reporter = ProgressReporter(run_dir)
    reporter.update(run_id=run_id, log_dir=str(run_dir))
    logger.info(f"run_id={run_id} log_dir={run_dir}")
    return Progress.console(reporter=reporter)
