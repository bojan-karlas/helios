"""HTML report of MIL training traces."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from helios.cli.main import app
from helios.components.mil import train_mil
from helios.components.mil_trace_report import COLORS, find_traces, render_trace_report
from tests.test_component_mil import _make_slide


def _run(tmp_path: Path, run_id: str, folds: int) -> Path:
    run_dir = tmp_path / "logs" / run_id
    slides = [_make_slide(tmp_path, f"{run_id}-S{i}", n_tiles=12, label=i % 2, seed=i) for i in range(6)]
    for fold in range(folds):
        train_mil(
            slides, max_epochs=2, val_frac=0.34, device="cpu",
            trace_dir=run_dir / "traces" / "mil" / f"fold={fold}", trace_slides=slides[:2],
        )
    return run_dir


def test_report_overlays_runs_and_folds_with_house_colors(tmp_path):
    first, second = _run(tmp_path, "run-a", 2), _run(tmp_path, "run-b", 1)
    traces = find_traces([tmp_path / "logs"])
    assert [(t.run_id, t.fold) for t in traces] == [("run-a", "0"), ("run-a", "1"), ("run-b", "0")]
    assert all(t.selected_epoch in (1, 2) for t in traces)

    page = render_trace_report([first, second], tmp_path / "report.html").read_text()
    for text in ("run-a fold=0", "run-a fold=1", "run-b fold=0", "Learning curves", "Test-fold AUPRC",
                 "Training stability", "Model internals", "Validation predictions", "Events"):
        assert text in page
    assert COLORS["red"] in page and COLORS["blue"] in page
    assert 'http-equiv="refresh"' not in page
    live = render_trace_report([first], tmp_path / "live.html", refresh_seconds=30).read_text()
    assert 'http-equiv="refresh" content="30"' in live and "run-b" not in live


def test_report_without_traces_and_cli(tmp_path):
    empty = render_trace_report([tmp_path], tmp_path / "empty.html").read_text()
    assert "No traces found yet." in empty
    run_dir = _run(tmp_path, "run-c", 1)
    output = tmp_path / "cli.html"
    result = CliRunner().invoke(app, ["plot", "traces", str(run_dir), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert "run-c fold=0" in output.read_text()


def test_report_describes_weight_averaged_runs(tmp_path):
    slides = [_make_slide(tmp_path, f"S{i}", n_tiles=12, label=i % 2, seed=i) for i in range(6)]
    run_dir = tmp_path / "logs" / "run-avg"
    train_mil(
        slides, max_epochs=3, val_frac=0, average_from_epoch=2, device="cpu",
        trace_dir=run_dir / "traces" / "mil" / "fold=0", trace_slides=slides[:2],
    )
    page = render_trace_report([run_dir], tmp_path / "avg.html").read_text()
    assert "average of epochs 2-3" in page and "running weight average" in page
