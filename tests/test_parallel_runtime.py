"""Exercise Ray workers with CPU tasks and a synthetic GPU reservation."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest

from helios.stages._ray import ParallelConfig, StageExecutionError, WorkContext, run_work_units


def _record(unit: int, ctx: WorkContext, *, root: Path) -> str:
    import json
    import time

    import ray

    time.sleep(0.05)
    (root / f"{unit}.json").write_text(json.dumps({
        "pid": os.getpid(),
        "resources": ray.get_runtime_context().get_assigned_resources(),
    }))
    ctx.progress.log(f"processed {unit}")
    if unit == -1:
        raise ValueError("intentional worker failure")
    if unit == -2:
        raise SystemExit(17)
    return "completed"


@pytest.fixture(scope="module")
def cluster():
    import ray

    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False, log_to_driver=False)
    try:
        assert ray.is_initialized()
        yield
    finally:
        ray.shutdown()


def test_serial_does_not_start_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    from helios.stages import _ray

    def forbidden(*args, **kwargs):
        raise AssertionError("Serial execution must not initialize Ray")

    monkeypatch.setattr(_ray, "ray_session", forbidden)
    seen = []
    assert run_work_units([1], lambda item, ctx: seen.append(os.getpid()), stage="serial") == ["completed"]
    assert seen == [os.getpid()]


@pytest.mark.parametrize("persistent", [False, True])
def test_opt_in_single_unit_runs_remotely(cluster, tmp_path: Path, persistent: bool) -> None:
    import json

    result = run_work_units(
        [0], partial(_record, root=tmp_path), stage="single",
        parallel=ParallelConfig(max_cpus=2), persistent_workers=persistent, num_cpus=2,
    )
    assert result == ["completed"]
    output = json.loads((tmp_path / "0.json").read_text())
    assert output["pid"] != os.getpid()
    assert output["resources"]["CPU"] == 2


def test_actor_pool_reuses_workers_and_bounds_concurrency(cluster, tmp_path: Path) -> None:
    import json

    result = run_work_units(
        list(range(6)), partial(_record, root=tmp_path), stage="reuse",
        parallel=ParallelConfig(max_cpus=2), persistent_workers=True, num_cpus=2,
    )
    assert result == ["completed"] * 6
    outputs = [json.loads((tmp_path / f"{i}.json").read_text()) for i in range(6)]
    assert len({out["pid"] for out in outputs}) == 1
    assert all(out["resources"]["CPU"] == 2 for out in outputs)


def test_gpu_reservation_uses_cluster_not_coordinator(cluster, tmp_path: Path, monkeypatch) -> None:
    import json

    from helios.stages import _ray

    monkeypatch.setattr(_ray, "available_gpu_count", lambda: 0)
    config = ParallelConfig(max_cpus=2)
    assert _ray.gpu_reservation("cuda", config) is None
    assert _ray.gpu_reservation("cpu", config) == 0
    run_work_units(
        list(range(3)), partial(_record, root=tmp_path), stage="gpu",
        parallel=config, persistent_workers=True, num_gpus=_ray.gpu_reservation("cuda", config),
    )
    outputs = [json.loads((tmp_path / f"{i}.json").read_text()) for i in range(3)]
    assert len({out["pid"] for out in outputs}) == 1
    assert all(out["resources"]["GPU"] == 1 for out in outputs)


def test_auto_gpu_reservation_falls_back_on_cpu_cluster(cluster, tmp_path: Path, monkeypatch) -> None:
    import json

    import ray

    resources = ray.cluster_resources()
    resources.pop("GPU", None)
    monkeypatch.setattr(ray, "cluster_resources", lambda: resources)
    run_work_units(
        [0], partial(_record, root=tmp_path), stage="cpu-fallback",
        parallel=ParallelConfig(max_cpus=2), persistent_workers=True, num_gpus=None,
    )
    output = json.loads((tmp_path / "0.json").read_text())
    assert output["resources"].get("GPU", 0) == 0


@pytest.mark.parametrize("persistent", [False, True])
def test_worker_failures_keep_context_and_finish_other_units(cluster, tmp_path: Path, persistent: bool) -> None:
    import json

    from helios.progress import Progress, ProgressReporter

    reporter = ProgressReporter(tmp_path / "logs")
    with pytest.raises(StageExecutionError) as raised:
        run_work_units(
            [-1, 0, 1], partial(_record, root=tmp_path), stage="fail",
            parallel=ParallelConfig(max_cpus=2), persistent_workers=persistent,
            progress=Progress(reporter=reporter),
        )
    assert (tmp_path / "1.json").exists()
    failure, = raised.value.failures
    assert failure.work_id == "-1"
    assert "ValueError: intentional worker failure" in failure.error
    assert "_record" in failure.traceback_text
    assert failure.worker_id.startswith("ray")
    assert failure.log_path is not None
    assert "intentional worker failure" in Path(failure.log_path).read_text()
    status = json.loads(reporter.status_path.read_text())
    assert status["counts"] == {
        "total": 3, "queued": 0, "running": 0, "completed": 2, "skipped": 0, "failed": 1,
    }
    assert status["active_workers"] == {}


def test_actor_fail_fast_stops_queue_and_releases_resources(cluster, tmp_path: Path) -> None:
    config = ParallelConfig(max_cpus=2)
    with pytest.raises(StageExecutionError):
        run_work_units(
            [-1, 0], partial(_record, root=tmp_path), stage="fail-fast",
            parallel=config, persistent_workers=True, num_cpus=2, continue_on_error=False,
        )
    assert not (tmp_path / "0.json").exists()
    assert run_work_units(
        [1], partial(_record, root=tmp_path), stage="after-failure",
        parallel=config, persistent_workers=True, num_cpus=2,
    ) == ["completed"]


def test_actor_death_is_reported_as_stage_failure(cluster, tmp_path: Path) -> None:
    with pytest.raises(StageExecutionError) as raised:
        run_work_units(
            [-2], partial(_record, root=tmp_path), stage="dead-worker",
            parallel=ParallelConfig(max_cpus=2), persistent_workers=True,
        )
    failure, = raised.value.failures
    assert failure.work_id == "-2"
    assert failure.worker_id == "ray-worker"
    assert failure.traceback_text


def test_dead_actor_is_replaced_so_remaining_units_still_run(cluster, tmp_path: Path) -> None:
    import json

    units = [-2, 0, 1, 2, 3]
    with pytest.raises(StageExecutionError) as raised:
        run_work_units(
            units, partial(_record, root=tmp_path), stage="respawn",
            parallel=ParallelConfig(max_cpus=2), persistent_workers=True, num_cpus=2,
        )
    assert [f.work_id for f in raised.value.failures] == ["-2"]
    outputs = [json.loads((tmp_path / f"{i}.json").read_text()) for i in (0, 1, 2, 3)]
    assert len({out["pid"] for out in outputs}) == 1, "a fresh actor took over the dead slot"


def test_status_mailbox_drops_oldest_non_lifecycle_events_and_counts() -> None:
    from helios.stages._ray import _StatusActor

    mailbox = _StatusActor(limit=3)
    mailbox.push({"kind": "start", "work_id": "a"})
    for i in range(5):
        mailbox.push({"kind": "log", "message": str(i)})
    mailbox.push({"kind": "start", "work_id": "b"})
    events, dropped = mailbox.drain()
    assert dropped == 3
    assert [e["kind"] for e in events] == ["start", "log", "log", "start"]
    assert [e["message"] for e in events if e["kind"] == "log"] == ["3", "4"]
    assert mailbox.drain() == ([], 0)


def test_impossible_reservation_fails_without_hanging(cluster) -> None:
    with pytest.raises(ValueError, match="cannot satisfy"):
        run_work_units(
            [0], lambda item, ctx: None, stage="resources", parallel=ParallelConfig(max_cpus=2), num_cpus=999999,
        )


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_work_units([1, 1], lambda item, ctx: None, stage="duplicates")


def test_real_mil_parallel_fit_and_prediction_match_serial(cluster, tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    from helios.data.io import ArtifactStore
    from helios.data.resolver import Resolver, Roots
    from helios.stages.mil import fit_mil, predict_mil
    from tests.fixtures import make_cohort

    root = make_cohort(tmp_path / "example", n_images=4, n_patients=4)
    store = ArtifactStore(Resolver(Roots.single(root)))
    ids = pd.read_csv(root / "metadata/image_metadata.csv")["image_id"].tolist()
    rng = np.random.default_rng(42)
    for image_id in ids:
        store.write(
            "tile_features", {"features": rng.normal(size=(3, 4)).astype(np.float32)},
            size_mm=0.25, model="virchow2", image_id=image_id,
        )
    store.write("splits", pd.DataFrame([
        {"image_id": image_id, "fold": fold, "split": "train" if (index // 2 == fold) else "test"}
        for fold in (0, 1) for index, image_id in enumerate(ids)
    ]))
    config = ParallelConfig(max_cpus=2)
    fit_mil(
        datasets=[root], parallel=config, target="disease_pfs_recurred",
        device="cpu", max_epochs=1, val_frac=0,
    )
    for fold in (0, 1):
        assert store.exists("model_mil", fold=fold)
    predict_mil(datasets=[root], device="cpu")
    expected = store.read("whole_image_risk_score_ensemble")
    expected_grad = store.read("aligned_slide_embedding_grad", fold=0, image_id=ids[0])
    predict_mil(datasets=[root], device="cpu", parallel=config, overwrite=True)
    pd.testing.assert_frame_equal(expected, store.read("whole_image_risk_score_ensemble"))
    np.testing.assert_allclose(
        expected_grad, store.read("aligned_slide_embedding_grad", fold=0, image_id=ids[0]), atol=1e-7,
    )
    path = store.path("aligned_slide_embedding", fold=0, image_id=ids[0])
    modified = path.stat().st_mtime_ns
    predict_mil(datasets=[root], device="cpu", parallel=config, image_ids=[ids[0]])
    assert path.stat().st_mtime_ns == modified
    pd.testing.assert_frame_equal(expected, store.read("whole_image_risk_score_ensemble"))
