"""Fold scheduler wiring for concept, morphology, and risk fitting."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from helios.data.cohort import ResolvedDataset
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages import concept, morphology, risk
from helios.stages._runtime import ParallelConfig, run_work_units
from tests.fixtures import make_cohort


@pytest.mark.parametrize(
    ("module", "fit_name", "worker_name", "stage", "fit_options"),
    [
        (concept, "fit_concept", "_fit_concept_fold", "fit concept", {}),
        (morphology, "fit_morphology", "_fit_morphology_fold", "fit morphology", {}),
        (risk, "fit_risk", "_fit_risk_fold", "fit risk clinical", {"risk": ["clinical"]}),
    ],
)
@pytest.mark.parametrize(
    ("parallel", "expected_continue_on_error"),
    [(None, False), (ParallelConfig(max_cpus=2), True)],
)
def test_fit_stages_submit_fold_units(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: ModuleType,
    fit_name: str,
    worker_name: str,
    stage: str,
    fit_options: dict[str, object],
    parallel: ParallelConfig | None,
    expected_continue_on_error: bool,
) -> None:
    resolved = [ResolvedDataset(tmp_path / "synthetic-set")]
    monkeypatch.setattr(module, "resolve_datasets", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(module, "folds_across", lambda *args, **kwargs: [0, 1])
    submitted: dict[str, object] = {}

    def capture(units, fn, **kwargs):
        submitted.update(units=list(units), fn=fn, **kwargs)

    monkeypatch.setattr(module, "run_work_units", capture)
    getattr(module, fit_name)(
        datasets=[resolved[0].root],
        parallel=parallel,
        **fit_options,
    )

    assert submitted["units"] == [0, 1]
    assert isinstance(submitted["fn"], partial)
    assert submitted["fn"].func is getattr(module, worker_name)
    assert submitted["stage"] == stage
    assert submitted["parallel"] == (parallel or ParallelConfig())
    assert submitted["num_cpus"] >= 1
    assert submitted["continue_on_error"] is expected_continue_on_error
    assert "progress" in submitted


def test_serial_work_unit_reraises_original_exception_type() -> None:
    class FitFailure(ValueError):
        pass

    def fail(_fold, _ctx):
        raise FitFailure("synthetic failure")

    with pytest.raises(FitFailure, match="synthetic failure") as exc_info:
        run_work_units(
            [0],
            fail,
            stage="fit synthetic",
            parallel=ParallelConfig(),
            continue_on_error=False,
        )

    assert type(exc_info.value) is FitFailure


def test_morphology_serial_fold_loads_inputs_and_writes_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = make_cohort(tmp_path / "synthetic-set", n_images=2, n_patients=2)
    store = ArtifactStore(Resolver(Roots.single(root)))
    image_ids = store.read("image_metadata")["image_id"].tolist()
    for image_id in image_ids:
        store.write(
            "aligned_patch_embeddings",
            {"embeddings": np.ones((2, 3), dtype=np.float32)},
            image_id=image_id,
        )
        store.write(
            "patch_attention",
            pd.DataFrame({"tile_id": [0, 1], "attention": [0.4, 0.6]}),
            image_id=image_id,
        )
    captured: dict[str, object] = {}

    def fake_train(paths, attention, labels, **kwargs):
        captured.update(paths=paths, attention=attention, labels=labels, kwargs=kwargs)
        return {"trained": True}

    monkeypatch.setattr(morphology, "train_morphology", fake_train)
    morphology.fit_morphology(datasets=[root], parallel=ParallelConfig())

    assert store.exists("model_morphology", fold=None)
    assert set(captured["paths"]) == set(image_ids)
    assert set(captured["attention"]["image_id"]) == set(image_ids)
    assert set(captured["labels"]["image_id"]) == set(image_ids)
