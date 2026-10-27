"""CLI logs follow the resolved results role, independently of model storage."""
import pytest
from typer.testing import CliRunner

from helios.cli._common import console_progress, resolved_results_root
from helios.cli.main import app
from helios.data.cohort import ResolvedDataset
from helios.data.resolver import StorageRoots


@pytest.mark.parametrize("explicit_results", [False, True])
def test_log_base_matches_results_role(tmp_path, explicit_results):
    ds = ResolvedDataset(tmp_path / "source", cohort_name="experiment")
    roots = StorageRoots(
        models_root=tmp_path / "models",
        results_root=tmp_path / "results" if explicit_results else None,
    )
    expected = tmp_path / "results/experiment" if explicit_results else ds.root / "results"
    assert resolved_results_root(ds, roots) == expected


def test_explicit_log_dir_overrides_default(tmp_path, monkeypatch):
    seen = []

    def create(base, tag):
        seen.append((base, tag))
        return "run", tmp_path / "run"

    class Reporter:
        def __init__(self, path):
            pass

        def update(self, **kwargs):
            pass

    import logging

    monkeypatch.setattr("helios.cli._common.create_run_log_dir", create)
    monkeypatch.setattr("helios.cli._common.configure_logging", lambda path: logging.getLogger("test"))
    monkeypatch.setattr("helios.cli._common.ProgressReporter", Reporter)
    console_progress(tmp_path / "results", log_dir=tmp_path / "custom", run_tag="example")
    assert seen == [(tmp_path / "custom", "example")]
    console_progress(tmp_path / "results")
    assert seen[-1] == (tmp_path / "results/logs", None)


def test_every_output_command_uses_resolved_results_root(monkeypatch, tmp_path):
    from importlib import import_module

    from helios.stages import STAGES

    runner = CliRunner()
    for verb, stages in STAGES.items():
        for noun, stage in stages.items():
            seen = []
            with monkeypatch.context() as patch:
                patch.setattr(f"{stage.__module__}.{stage.__name__}", lambda **kwargs: None)
                patch.setattr("helios.config.stage_config.load_stage_params", lambda *a, **k: {})
                module = import_module(f"helios.cli.{'main' if verb == 'report' else verb}")
                patch.setattr(module, "console_progress", lambda path, _seen=seen, **kwargs: _seen.append(path))
                command = [verb] if verb == "report" else [verb, noun]
                result = runner.invoke(app, command + [
                    "--dataset", str(tmp_path / "source"), "--models-root", str(tmp_path / "models"),
                ])
                assert result.exit_code == 0, (command, result.exception)
            assert seen == [tmp_path / "source/results"]


def test_bare_prep_uses_resolved_results_root(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("helios.stages.STAGES", {"prep": {}})
    monkeypatch.setattr("helios.cli.prep.console_progress", lambda path, **kw: seen.append(path))
    result = CliRunner().invoke(app, ["prep", "--dataset", str(tmp_path / "source")])
    assert result.exit_code == 0, result.exception
    assert seen == [tmp_path / "source/results"]
