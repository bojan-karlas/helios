"""Training feedback timing, terminal output, and log-only operation."""
from __future__ import annotations

from io import StringIO

import pytest

from helios.progress import Progress
from helios.progress.training import TrainingProgress, duration


class Sink:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def test_full_budget_eta_accounts_for_current_epoch(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("helios.progress.training.monotonic", lambda: now[0])
    with TrainingProgress(Progress.null(), 4, "[fold=2] ").display() as feedback:
        feedback.start_epoch()
        assert feedback.budget_eta() == "estimating"
        now[0] = 100
        assert feedback.finish_epoch() == 100
        assert feedback.budget_eta() == "00:05:00"
        feedback.start_epoch()
        now[0] = 140
        assert feedback.budget_eta() == "00:04:20"
        now[0] = 220
        feedback.finish_epoch()
        assert feedback.budget_eta() == "00:03:40"


def test_periodic_feedback_counts_skips_loss_and_reports_eta(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("helios.progress.training.monotonic", lambda: now[0])
    sink = Sink()
    with TrainingProgress(Progress(reporter=sink, use_tqdm=False), 3, "[fold=1] ").display() as feedback:
        feedback.start_epoch()
        for item in feedback.phase([0, 1, 2, 3], "train"):
            now[0] += 15
            if item == 1:
                feedback.skipped += 1
            else:
                feedback.record_loss(0.5)
        assert feedback.steps == 3
        assert feedback.skipped == 1
        assert feedback.mean_loss == "0.5000"
        feedback.finish_epoch()
    messages = [fields["message"] for kind, fields in sink.events if kind == "log"]
    assert any("[fold=1] Epoch 1/3 train 2/4 slides" in m and "phase_ETA=00:00:30" in m for m in messages)
    assert any("4/4 slides" in m and "skipped=1" in m and "loss=0.5000" in m for m in messages)
    assert not any("train 1/4 slides" in m or "train 3/4 slides" in m for m in messages)


@pytest.mark.parametrize("interactive", [False, True])
def test_terminal_bars_and_redirected_output(monkeypatch, interactive):
    class Output(StringIO):
        def isatty(self):
            return interactive

    output = Output()
    monkeypatch.setattr("sys.stderr", output)
    with TrainingProgress(Progress(), 1, "").display() as feedback:
        feedback.start_epoch()
        for _ in feedback.phase([1, 2], "train"):
            feedback.record_loss(0.2)
        for _ in feedback.phase([1], "valid"):
            pass
        feedback.finish_epoch()
    text = output.getvalue()
    if interactive:
        assert "Epochs" in text
        assert "Epoch 1/1 train" in text
        assert "Epoch 1/1 valid" in text
        assert "ETA to 1 epochs=" in text
        assert "100%" in text
        phase_lines = [
            line for line in text.split("\r")
            if "Epoch 1/1 train" in line or "Epoch 1/1 valid" in line
        ]
        assert phase_lines
        assert all("ETA to" not in line for line in phase_lines)
    else:
        assert text == ""


def test_null_progress_emits_nothing_even_in_terminal(monkeypatch):
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    sink = Sink()
    progress = Progress(reporter=sink, enabled=False)
    with TrainingProgress(progress, 1, "").display() as feedback:
        feedback.start_epoch()
        assert list(feedback.phase([1], "train")) == [1]
        feedback.finish_epoch()
    assert sink.events == []


def test_outer_stage_bar_is_silent_when_redirected(monkeypatch):
    output = StringIO()
    sink = Sink()
    monkeypatch.setattr("sys.stderr", output)
    assert list(Progress(reporter=sink).task([1, 2], desc="fit mil")) == [1, 2]
    assert output.getvalue() == ""
    assert sink.events == [
        ("task_start", {"desc": "fit mil", "total": 2}),
        ("task_done", {"desc": "fit mil", "count": 2}),
    ]


@pytest.mark.parametrize(("seconds", "expected"), [(None, "estimating"), (3661, "01:01:01"), (-1, "00:00:00")])
def test_duration(seconds, expected):
    assert duration(seconds) == expected
