"""Cohort-definition resolution tests.

Covers the ``--dataset`` resolution layer: plain roots, cohort YAML files (union
of roots + filter), ``ResolvedDataset`` pass-through/idempotency, and filter
composition.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from helios.data.cohort import (
    ResolvedDataset,
    combine_filters,
    is_cohort_file,
    load_cohort_spec,
    resolve_datasets,
)


def test_plain_root_is_single_unfiltered_dataset() -> None:
    resolved = resolve_datasets(["/data/MGB"])
    assert resolved == [ResolvedDataset(Path("/data/MGB"), where=None)]


def test_adhoc_where_applies_to_every_root() -> None:
    resolved = resolve_datasets(["/data/MGB", "/data/MRV"], where="path_stage == 'I'")
    assert [d.where for d in resolved] == ["path_stage == 'I'", "path_stage == 'I'"]


def test_resolve_is_idempotent_on_resolved_dataset() -> None:
    once = resolve_datasets(["/data/MGB"])
    twice = resolve_datasets(once)
    assert once == twice


def test_resolve_combines_existing_and_adhoc_filter() -> None:
    ds = ResolvedDataset(Path("/data/MGB"), where="a == 1")
    (out,) = resolve_datasets([ds], where="b == 2")
    assert out.where == "(a == 1) and (b == 2)"


def test_cohort_file_expands_to_member_roots_with_filter(tmp_path: Path) -> None:
    spec = {
        "name": "train",
        "datasets": ["/data/MGB", "sibling"],
        "filter": "path_stage in ['I', 'II']",
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(spec))

    resolved = resolve_datasets([path])

    assert [d.root for d in resolved] == [Path("/data/MGB"), tmp_path / "sibling"]
    assert all(d.where == "path_stage in ['I', 'II']" for d in resolved)


def test_cohort_file_filter_and_adhoc_filter_compose(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"datasets": ["/data/MGB"], "filter": "a == 1"}))

    (out,) = resolve_datasets([path], where="b == 2")
    assert out.where == "(a == 1) and (b == 2)"


def test_load_cohort_spec_resolves_relative_roots(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"datasets": ["../peer", "/abs"]}))

    spec = load_cohort_spec(path)
    assert Path(spec.datasets[0]).resolve() == (tmp_path.parent / "peer").resolve()
    assert spec.datasets[1] == "/abs"


def test_is_cohort_file_distinguishes_yaml_from_root(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("datasets: [/data/MGB]\n")
    assert is_cohort_file(yaml_file)
    assert not is_cohort_file(tmp_path)
    assert not is_cohort_file(ResolvedDataset(Path("/data/MGB")))


def test_combine_filters_identity_and_and() -> None:
    assert combine_filters(None, None) is None
    assert combine_filters("a", None) == "a"
    assert combine_filters(None, "b") == "b"
    assert combine_filters("a", "b") == "(a) and (b)"
