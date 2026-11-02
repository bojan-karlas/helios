"""Explicit directory selection and console-only CLI operation."""
import pytest
from typer.testing import CliRunner

from helios.cli._common import console_progress, prepare_invocation
from helios.cli.main import app
from helios.stages._runtime import cohort_store


def test_console_only_progress_creates_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    progress = console_progress()
    progress.log("visible")
    assert progress.reporter is None
    assert "visible" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_explicit_logs_directory(tmp_path):
    progress = console_progress(logs_dir=tmp_path / "logs", run_tag="example")
    progress.log("saved")
    paths = list((tmp_path / "logs").glob("*/run.log"))
    assert len(paths) == 1
    assert "saved" in paths[0].read_text()


def test_output_defaults_do_not_redirect_inputs(tmp_path):
    invocation = prepare_invocation(
        [str(tmp_path / "source")], output_dir=tmp_path / "experiment",
    )
    store = cohort_store(invocation.datasets[0], invocation.dirs)
    assert store.path("thumbnail", image_id="slide") == tmp_path / "source/preprocessing/thumbnails/slide.png"
    assert store.output_path("thumbnail", image_id="slide") is None
    assert store.output_path("model_mil", fold=None) == tmp_path / "experiment/models/mil/fold=None"
    with pytest.raises(ValueError, match="explicit --models-dir"):
        store.path("model_mil", fold=None)
    assert invocation.logs_dir == tmp_path / "experiment/logs"


def test_role_directories_override_output_defaults(tmp_path):
    invocation = prepare_invocation(
        [str(tmp_path / "source")], output_dir=tmp_path / "experiment",
        models_dir=tmp_path / "models", results_dir=tmp_path / "results",
        preprocessing_dir=tmp_path / "pre", logs_dir=tmp_path / "logs",
        splits_file=tmp_path / "assignments.csv", where="path_thickness > 1",
        ray_address="auto", max_cpus=3,
    )
    store = cohort_store(invocation.datasets[0], invocation.dirs)
    assert store.output_resolver.dirs.models_dir == tmp_path / "models"
    assert store.output_resolver.dirs.results_dir == tmp_path / "results/source"
    assert store.resolver.dirs.preprocessing_dir == tmp_path / "pre"
    assert invocation.logs_dir == tmp_path / "logs"
    assert invocation.parallel.max_cpus == 3
    assert invocation.datasets[0].where == "path_thickness > 1"
    assert invocation.dirs.splits_file == tmp_path / "assignments.csv"


def test_every_output_command_accepts_directory_options(monkeypatch, tmp_path):
    from helios.stages import STAGES

    for verb, stages in STAGES.items():
        for noun, stage in stages.items():
            seen = []
            with monkeypatch.context() as patch:
                patch.setattr(
                    f"{stage.__module__}.{stage.__name__}", lambda _seen=seen, **kwargs: _seen.append(kwargs),
                )
                patch.setattr("helios.config.stage_config.load_stage_params", lambda *a, **k: {})
                command = [verb] if verb == "report" else [verb, noun]
                result = CliRunner().invoke(app, command + [
                    "--dataset", str(tmp_path / "source"),
                    "--output-dir", str(tmp_path / "experiment"),
                    "--models-dir", str(tmp_path / "trained-models"),
                    "--preprocessing-dir", str(tmp_path / "pre"),
                ])
                assert result.exit_code == 0, (command, result.exception, result.output)
            assert seen[0]["dirs"].output_dir == tmp_path / "experiment"
            assert seen[0]["dirs"].models_dir == tmp_path / "trained-models"
            assert not (tmp_path / "source").exists()


def test_bare_prep_rejects_unsaved_intermediate_artifacts(tmp_path):
    result = CliRunner().invoke(app, [
        "prep", "--dataset", str(tmp_path / "source"), "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0
    assert "Full prep requires" in result.output
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("old_option", ["--models-root", "--results-root", "--preprocessing-root", "--log-dir"])
def test_old_options_are_not_aliases(tmp_path, old_option):
    result = CliRunner().invoke(app, [
        "fit", "mil", "--dataset", str(tmp_path / "source"), old_option, str(tmp_path / "out"),
    ])
    assert result.exit_code != 0
    assert "No such option" in result.output
