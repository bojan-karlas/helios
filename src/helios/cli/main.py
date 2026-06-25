"""HELIOS command-line interface (thin Typer wrappers over the library).

Every command mirrors an importable function; the CLI adds no logic beyond
config loading and printing a short summary. ``helios`` is the entry point
(see pyproject ``[project.scripts]``).
"""
from __future__ import annotations

from pathlib import Path

import typer

from helios.config.loader import load_config
from helios.pipeline.plan import infer_plan, run_infer, run_train, train_plan

app = typer.Typer(add_completion=False, help="HELIOS computational-pathology pipeline.")


def _summarize(results) -> None:
    from collections import Counter

    counts = Counter(r.status for r in results)
    typer.echo("stage outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


@app.command()
def fit(
    config: Path | None = typer.Option(None, "--config", "-c", help="Run config YAML."),
    base: Path | None = typer.Option(None, "--base", help="Override paths.base."),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if outputs exist."),
) -> None:
    """Train the full pipeline (hand-wired train plan)."""
    overrides = {"paths": {"base": str(base)}} if base else None
    cfg = load_config(config, overrides)
    results = run_train(cfg, force=force)
    _summarize(results)


@app.command()
def predict(
    config: Path | None = typer.Option(None, "--config", "-c", help="Run config YAML."),
    base: Path | None = typer.Option(None, "--base", help="Override paths.base."),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if outputs exist."),
) -> None:
    """Run inference (hand-wired infer plan)."""
    overrides = {"paths": {"base": str(base)}} if base else None
    cfg = load_config(config, overrides)
    results = run_infer(cfg, force=force)
    _summarize(results)


@app.command()
def plan(mode: str = typer.Argument("train", help="train | infer")) -> None:
    """Print the ordered component plan for a mode."""
    stages = train_plan() if mode == "train" else infer_plan()
    for i, cid in enumerate(stages, 1):
        typer.echo(f"{i:>2}. {cid}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
