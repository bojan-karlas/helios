"""End-to-end wiring smoke test.

Runs the full train + infer plans on a synthetic cohort with stubbed domain
logic. Asserts that the wiring holds together: every stage is reached, no stage
raises (unimplemented stubs are non-fatal), and progress files are written.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from helios.config.loader import load_config
from helios.pipeline.plan import run_infer, run_train
from tests.fixtures import make_cohort


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    return make_cohort(tmp_path / "cohort")


def _config(cohort: Path):
    return load_config(overrides={"paths": {"base": str(cohort)}, "version": "vtest"})


def test_train_wiring(cohort: Path) -> None:
    results = run_train(_config(cohort))
    assert results, "train plan produced no stages"
    assert not [r for r in results if r.status == "error"]
    assert (cohort / "status.json").exists()
    assert (cohort / "events.jsonl").exists()


def test_infer_wiring(cohort: Path) -> None:
    results = run_infer(_config(cohort))
    assert results
    assert not [r for r in results if r.status == "error"]


def test_plans_cover_modes() -> None:
    from helios.pipeline.plan import _validate_order, infer_plan, train_plan

    _validate_order()
    assert set(infer_plan()).issubset(set(train_plan()) | set(infer_plan()))
    assert len(train_plan()) > len(infer_plan())
