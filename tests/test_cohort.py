"""Cohort-definition resolution tests.

Covers the ``--dataset`` resolution layer: plain roots, cohort YAML files (union
of roots + filter), ``ResolvedDataset`` pass-through/idempotency, and filter
composition.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helios.data.cohort import (
    ResolvedDataset,
    combine_filters,
    is_cohort_file,
    load_cohort_spec,
    resolve_datasets,
)


def test_plain_root_is_single_unfiltered_dataset() -> None:
    resolved = resolve_datasets(["/data/COHORT1"])
    assert resolved == [ResolvedDataset(Path("/data/COHORT1"), where=None)]


def test_adhoc_where_applies_to_every_root() -> None:
    resolved = resolve_datasets(["/data/COHORT1", "/data/COHORT2"], where="path_stage == 'I'")
    assert [d.where for d in resolved] == ["path_stage == 'I'", "path_stage == 'I'"]


def test_resolve_is_idempotent_on_resolved_dataset() -> None:
    once = resolve_datasets(["/data/COHORT1"])
    twice = resolve_datasets(once)
    assert once == twice


def test_resolve_combines_existing_and_adhoc_filter() -> None:
    ds = ResolvedDataset(Path("/data/COHORT1"), where="a == 1")
    (out,) = resolve_datasets([ds], where="b == 2")
    assert out.where == "(a == 1) and (b == 2)"


def test_cohort_file_expands_to_member_roots_with_filter(tmp_path: Path) -> None:
    spec = {
        "name": "TRAIN_COHORT",
        "datasets": ["/data/COHORT1", "sibling"],
        "filter": "path_stage in ['I', 'II']",
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(spec))

    resolved = resolve_datasets([path])

    assert [d.root for d in resolved] == [Path("/data/COHORT1"), tmp_path / "sibling"]
    assert all(d.where == "path_stage in ['I', 'II']" for d in resolved)


def test_cohort_file_filter_and_adhoc_filter_compose(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"datasets": ["/data/COHORT1"], "filter": "a == 1"}))

    (out,) = resolve_datasets([path], where="b == 2")
    assert out.where == "(a == 1) and (b == 2)"


def test_load_cohort_spec_resolves_relative_roots(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"datasets": ["../peer", "/abs"]}))

    spec = load_cohort_spec(path)
    assert spec.datasets[0].root.resolve() == (tmp_path.parent / "peer").resolve()
    assert spec.datasets[1].root == Path("/abs")


def test_is_cohort_file_distinguishes_yaml_from_root(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("datasets: [/data/COHORT1]\n")
    assert is_cohort_file(yaml_file)
    assert not is_cohort_file(tmp_path)
    assert not is_cohort_file(ResolvedDataset(Path("/data/COHORT1")))


def test_combine_filters_identity_and_and() -> None:
    assert combine_filters(None, None) is None
    assert combine_filters("a", None) == "a"
    assert combine_filters(None, "b") == "b"
    assert combine_filters("a", "b") == "(a) and (b)"


def test_extended_entries_preserve_constituent_storage_and_filters(tmp_path: Path) -> None:
    path = tmp_path / "union.yaml"
    path.write_text(yaml.safe_dump({
        "name": "union",
        "filter": "a == 1",
        "datasets": [
            {"source_root": "A", "preprocessing_root": "pre/A", "features_root": "/features/A"},
            "B",
        ],
    }))
    a, b = resolve_datasets([path], where="b == 2")
    assert a.root == tmp_path / "A"
    assert a.preprocessing_root == tmp_path / "pre/A"
    assert a.features_root == Path("/features/A")
    assert b.root == tmp_path / "B"
    assert b.preprocessing_root is None and b.features_root is None
    assert a.where == b.where == "(a == 1) and (b == 2)"
    assert resolve_datasets([a, b]) == [a, b]


@pytest.mark.parametrize("contents", [
    "datasets: /data/A",
    "datasets: []",
    "datasets: [{features_root: /features}]",
    "datasets: [{source_root: /data/A, feature_root: /features}]",
    "datasets: [42]",
    "datasets: [/data/A]\nfilter: 42",
])
def test_invalid_manifests_report_errors(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_cohort_spec(path)


def test_roots_alias_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("roots: [/data/A]\n")
    (resolved,) = resolve_datasets([path])
    assert resolved.root == Path("/data/A")
    assert resolved.cohort_name == "old"
    assert resolved.cohort_file == path.resolve()


def test_cli_forwards_storage_roots_to_every_stage(monkeypatch, tmp_path: Path) -> None:
    from importlib import import_module

    from typer.testing import CliRunner

    from helios.cli.main import app
    from helios.data.resolver import StorageRoots
    from helios.stages import STAGES

    roots = StorageRoots(
        preprocessing_root=tmp_path / "pre",
        features_root=tmp_path / "features",
        models_root=tmp_path / "models",
        results_root=tmp_path / "results",
    )
    options = [
        "--dataset", str(tmp_path / "source"),
        "--preprocessing-root", str(roots.preprocessing_root),
        "--features-root", str(roots.features_root),
        "--models-root", str(roots.models_root),
        "--results-root", str(roots.results_root),
    ]
    monkeypatch.setattr("helios.config.stage_config.load_stage_params", lambda *args, **kwargs: {})
    runner = CliRunner()
    for verb, stages in STAGES.items():
        for noun, stage in stages.items():
            seen = []
            with monkeypatch.context() as patch:
                patch.setattr(f"{stage.__module__}.{stage.__name__}", lambda _seen=seen, **kwargs: _seen.append(kwargs))
                module = import_module(f"helios.cli.{'main' if verb == 'report' else verb}")
                patch.setattr(module, "console_progress", lambda path, **kw: None)
                command = [verb] if verb == "report" else [verb, noun]
                result = runner.invoke(app, command + options)
            assert result.exit_code == 0, (command, result.output, result.exception)
            assert len(seen) == 1
            assert seen[0]["roots"] == roots
            assert seen[0]["datasets"] == [ResolvedDataset(tmp_path / "source")]


def test_cli_rejects_old_output_root_with_migration_guidance(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from helios.cli.main import app

    result = CliRunner().invoke(
        app, ["fit", "mil", "--dataset", str(tmp_path), "--output-root", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "--output-root has been removed" in result.output
    assert "--models-root" in result.output
    assert not (tmp_path / "out").exists()
