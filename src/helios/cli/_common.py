"""Shared helpers for the verb sub-apps (``prep``, ``fit``, ``predict``, ``report``).

Each verb is a Typer sub-app whose commands are the nouns. A command is a thin
wrapper: it resolves effective parameters (defaults + ``--config`` + flags),
builds a console Progress backed by a ProgressReporter, and calls the importable
stage function. Comma-separated list flags REPLACE the configured list
(consistent with the config precedence rules).
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.progress import Progress, configure_logging
from helios.progress.reporter import ProgressReporter


def reject_output_root(value: Path | None) -> None:
    if value is not None:
        raise typer.BadParameter(
            "--output-root has been removed. Use --preprocessing-root, --features-root, "
            "--models-root, and/or --results-root to select artifact destinations."
        )


PREPROCESSING_ROOT = typer.Option(None, "--preprocessing-root", help="Preprocessing base; single dataset only.")
FEATURES_ROOT = typer.Option(None, "--features-root", help="Feature-store base; single dataset only.")
MODELS_ROOT = typer.Option(None, "--models-root", help="Shared model base; required for pooled training.")
RESULTS_ROOT = typer.Option(None, "--results-root", help="Results base; artifacts are grouped by dataset name.")
REMOVED_OUTPUT_ROOT = typer.Option(None, "--output-root", hidden=True, callback=reject_output_root)


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


def console_progress(output_root: str | Path) -> Progress:
    """A console Progress that also writes status.json / events.jsonl / run.log."""
    configure_logging(Path(output_root))
    return Progress.console(reporter=ProgressReporter(Path(output_root)))
