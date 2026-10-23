"""Ray parallel execution, centralized worker status reporting, and failure aggregation.

Execution is serial and single-process by default: :class:`ParallelConfig` only
enables Ray when attaching to an existing cluster (``ray_address``) or when more
than one CPU is requested (``max_cpus > 1``).

- Without ``ray_address``, a local Ray runtime is started respecting
  ``CUDA_VISIBLE_DEVICES`` and ``max_cpus``, and shut down on exit.
- With ``ray_address``, the existing cluster is attached and left running on
  exit so bash scripts or services can share the pool across sequential or
  concurrent commands.

Workers never touch the coordinator's ``status.json`` / ``events.jsonl``: they
write their own ``workers/<worker-id>.log`` and forward log / lifecycle events
to a status actor that the coordinator drains while it renders one console
progress display. Ray task results are authoritative for completion.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from helios.progress import DEFAULT_PROGRESS, Progress, ProgressReporter, worker_log

T = TypeVar("T")

WorkStatus = Literal["queued", "running", "completed", "skipped", "failed"]


@dataclass(frozen=True)
class ParallelConfig:
    """Runtime parallelism passed from the CLI (``--ray-address``, ``--max-cpus``)."""

    ray_address: str | None = None
    max_cpus: int | None = None

    @property
    def enabled(self) -> bool:
        """Whether work units run on Ray instead of inline in this process."""
        return self.ray_address is not None or (self.max_cpus or 0) > 1


@dataclass(frozen=True)
class WorkContext:
    """What a work unit receives besides its item: a progress sink and its worker identity."""

    progress: Progress
    worker_id: str
    log_path: Path | None = None

    @property
    def log_path_str(self) -> str | None:
        return str(self.log_path) if self.log_path is not None else None


@dataclass(frozen=True)
class WorkResult:
    """Outcome of one work unit, returned (never raised) by workers."""

    work_id: str
    worker_id: str
    status: WorkStatus
    log_path: str | None = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True)
class WorkFailure:
    """Record of one failed work unit."""

    work_id: str
    worker_id: str
    error: str
    traceback_text: str
    log_path: str | None = None

    @classmethod
    def from_result(cls, result: WorkResult) -> WorkFailure:
        return cls(
            work_id=result.work_id,
            worker_id=result.worker_id,
            error=result.error or "Task failed",
            traceback_text=result.traceback_text or "",
            log_path=result.log_path,
        )


class StageExecutionError(RuntimeError):
    """Raised at the end of a stage when one or more work units failed."""

    def __init__(self, stage: str, failures: Sequence[WorkFailure]) -> None:
        self.stage = stage
        self.failures = tuple(failures)
        summary = "; ".join(f"{f.work_id}: {f.error}" for f in self.failures[:5])
        super().__init__(f"{stage} had {len(self.failures)} failed work unit(s): {summary}")


@dataclass
class RunStatusTracker:
    """Coordinator-owned per-unit state machine (queued -> running -> terminal).

    Counts are derived from unit states, so late or duplicate worker events can
    never leave the summary inconsistent with the authoritative task results.
    """

    stage: str
    work_ids: Sequence[str]
    states: dict[str, WorkStatus] = field(init=False)
    active_workers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = {wid: "queued" for wid in self.work_ids}

    def on_start(self, work_id: str, worker_id: str, log_path: str | None, ts: float | None = None) -> None:
        if self.states.get(work_id) == "queued":
            self.states[work_id] = "running"
        self.active_workers[worker_id] = {
            "work_id": work_id,
            "detail": "running",
            "log_path": log_path,
            "ts": ts or time.time(),
        }

    def on_progress(self, worker_id: str, detail: str, ts: float | None = None) -> None:
        if worker_id in self.active_workers:
            self.active_workers[worker_id].update(detail=detail, ts=ts or time.time())

    def on_result(self, result: WorkResult) -> None:
        self.states[result.work_id] = result.status
        active = self.active_workers.get(result.worker_id)
        if active is not None and active["work_id"] == result.work_id:
            del self.active_workers[result.worker_id]

    def snapshot(self) -> dict[str, Any]:
        counts = {status: 0 for status in ("queued", "running", "completed", "skipped", "failed")}
        for status in self.states.values():
            counts[status] += 1
        return {
            "stage": self.stage,
            "counts": {"total": len(self.states), **counts},
            "active_workers": dict(self.active_workers),
        }


def available_cpu_count(max_cpus: int | None = None) -> int:
    """Available CPU budget respecting OS affinity, SLURM, and optional cap."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if slurm_cpus:
        try:
            cpus = min(cpus, int(slurm_cpus.split("(")[0]))
        except ValueError:
            pass
    if max_cpus is not None and max_cpus > 0:
        cpus = min(cpus, int(max_cpus))
    return max(1, cpus)


def available_gpu_count() -> int:
    """Available GPU count respecting CUDA_VISIBLE_DEVICES and torch.cuda."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        vis = vis.strip()
        if not vis or vis in {"-1", "none", "NoDevFiles"}:
            return 0
        return len([x for x in vis.split(",") if x.strip()])
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


@contextmanager
def ray_session(config: ParallelConfig | None = None) -> Iterator[bool]:
    """Manage a Ray connection: start/stop local Ray or attach without stopping."""
    cfg = config or ParallelConfig()
    if not cfg.enabled:
        yield False
        return

    import ray

    started_locally = False
    if not ray.is_initialized():
        if cfg.ray_address:
            ray.init(address=cfg.ray_address, ignore_reinit_error=True, log_to_driver=False)
        else:
            ray.init(
                num_cpus=available_cpu_count(cfg.max_cpus),
                num_gpus=available_gpu_count(),
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
            started_locally = True
    try:
        yield True
    finally:
        if started_locally and ray.is_initialized():
            ray.shutdown()


def _level(name: object) -> int:
    level = logging.getLevelName(str(name or "INFO"))
    return level if isinstance(level, int) else logging.INFO


def worker_identity(prefix: str = "worker") -> str:
    host = socket.gethostname().split(".")[0]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_tag = f"-gpu{vis}" if vis else ""
    return f"{prefix}-{host}-{os.getpid()}{gpu_tag}"


class _StatusActor:
    """Event mailbox: workers push, the coordinator drains (wrapped by ``ray.remote`` lazily)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def push(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def drain(self) -> list[dict[str, Any]]:
        out, self.events = self.events, []
        return out


class _ActorSink:
    """A :class:`~helios.progress.EventSink` forwarding worker events to the status actor."""

    def __init__(self, handle: Any, worker_id: str, work_id: str) -> None:
        self.handle = handle
        self.worker_id = worker_id
        self.work_id = work_id

    def event(self, kind: str, **fields: Any) -> None:
        self.handle.push.remote(
            {"kind": kind, "worker_id": self.worker_id, "work_id": self.work_id, "ts": time.time(), **fields}
        )


def _execute_unit(
    fn: Callable[[T, WorkContext], str | None],
    item: T,
    work_id: str,
    stage: str,
    log_dir: str | None,
    actor_handle: Any,
    num_cpus: float,
) -> WorkResult:
    """Ray task body: run one unit with a worker-local Progress and return a WorkResult."""
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(num_cpus))))
    worker_id = worker_identity("ray")
    with worker_log(log_dir, worker_id) as (logger, log_path):
        sink = _ActorSink(actor_handle, worker_id, work_id)
        ctx = WorkContext(Progress(reporter=sink, use_tqdm=False, logger=logger), worker_id, log_path)
        sink.event("start", log_path=ctx.log_path_str)
        logger.info(f"START stage={stage} work_id={work_id} host={socket.gethostname()} pid={os.getpid()}")
        try:
            status = fn(item, ctx) or "completed"
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"FAILED stage={stage} work_id={work_id}\n{tb}")
            return WorkResult(work_id, worker_id, "failed", ctx.log_path_str, f"{type(exc).__name__}: {exc}", tb)
        logger.info(f"DONE stage={stage} work_id={work_id} status={status}")
        return WorkResult(work_id, worker_id, status, ctx.log_path_str)  # type: ignore[arg-type]


def run_work_units(
    units: Sequence[T],
    fn: Callable[[T, WorkContext], str | None],
    *,
    stage: str,
    unit_id: Callable[[T], str] = str,
    progress: Progress = DEFAULT_PROGRESS,
    parallel: ParallelConfig | None = None,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
    continue_on_error: bool = True,
) -> list[WorkStatus]:
    """Execute ``units`` inline or across Ray tasks with centralized status & error reporting.

    ``fn(unit, ctx)`` returns ``"skipped"`` when the unit was skipped, or
    ``"completed"`` (or ``None``) when computed; ``ctx.progress`` is the sink the
    unit should log through. Unit ids must be unique within one call. Failed
    units are collected and raised together as :class:`StageExecutionError`
    after the remaining units finish (unless ``continue_on_error`` is false).
    """
    cfg = parallel or ParallelConfig()
    work_ids = [unit_id(u) for u in units]
    tracker = RunStatusTracker(stage, work_ids)
    reporter = progress.reporter if isinstance(progress.reporter, ProgressReporter) else None
    run_log_dir = reporter.out_dir if reporter is not None else None
    failures: list[WorkFailure] = []
    results: list[WorkStatus] = ["queued"] * len(units)

    def publish(status: str | None = None) -> None:
        if reporter is None:
            return
        if status is None:
            reporter.update(**tracker.snapshot())
        else:
            reporter.stage(stage, status, **tracker.snapshot())

    def record(idx: int, result: WorkResult) -> None:
        tracker.on_result(result)
        results[idx] = result.status
        if result.status == "failed":
            failures.append(WorkFailure.from_result(result))
            progress.log(
                f"FAILED {stage} work_id={result.work_id} worker={result.worker_id} "
                f"log={result.log_path}: {result.error}",
                level=logging.ERROR,
            )
        publish()

    def finish() -> list[WorkStatus]:
        publish("failed" if failures else "completed")
        if failures:
            raise StageExecutionError(stage, failures)
        return results

    publish("running")

    if not cfg.enabled or len(units) <= 1:
        worker_id = worker_identity("local")
        with worker_log(run_log_dir, worker_id) as (logger, log_path):
            ctx = WorkContext(progress, worker_id, log_path)
            for idx, (item, wid) in enumerate(zip(progress.task(units, desc=stage), work_ids, strict=True)):
                tracker.on_start(wid, worker_id, ctx.log_path_str)
                publish()
                logger.info(f"START stage={stage} work_id={wid}")
                try:
                    status = fn(item, ctx) or "completed"
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error(f"FAILED stage={stage} work_id={wid}\n{tb}")
                    record(
                        idx, WorkResult(wid, worker_id, "failed", ctx.log_path_str, f"{type(exc).__name__}: {exc}", tb)
                    )
                    if not continue_on_error:
                        publish("failed")
                        raise
                    continue
                logger.info(f"DONE stage={stage} work_id={wid} status={status}")
                record(idx, WorkResult(wid, worker_id, status, ctx.log_path_str))  # type: ignore[arg-type]
        return finish()

    with ray_session(cfg):
        import ray

        actor: Any = ray.remote(_StatusActor).options(num_cpus=0).remote()
        remote: Any = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, max_retries=0)(_execute_unit)
        log_dir_str = str(run_log_dir) if run_log_dir is not None else None
        refs = {
            remote.remote(fn, item, wid, stage, log_dir_str, actor, num_cpus): (idx, wid)
            for idx, (item, wid) in enumerate(zip(units, work_ids, strict=True))
        }

        def drain() -> None:
            for ev in ray.get(actor.drain.remote()):
                kind = ev.get("kind")
                worker_id = str(ev.get("worker_id", ""))
                if kind == "start":
                    tracker.on_start(str(ev.get("work_id", "")), worker_id, ev.get("log_path"), ev.get("ts"))
                elif kind == "log":
                    progress.log(f"[{ev.get('work_id')}] {ev.get('message', '')}", level=_level(ev.get("level")))
                else:
                    if kind == "task_start":
                        tracker.on_progress(worker_id, str(ev.get("desc", "running")), ev.get("ts"))
                    if reporter is not None:
                        reporter.event(kind, **{k: v for k, v in ev.items() if k not in {"kind", "ts"}})
            publish()

        pending = list(refs)
        for _ in progress.task(range(len(units)), desc=stage):
            done: list[Any] = []
            while not done:
                done, pending = ray.wait(pending, num_returns=1, timeout=0.5)
                drain()
            idx, wid = refs[done[0]]
            try:
                result = ray.get(done[0])
            except Exception as exc:
                # The task body returns failures; reaching here means the worker itself died.
                first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                result = WorkResult(wid, "ray-worker", "failed", None, f"{type(exc).__name__}: {first}", str(exc))
            record(idx, result)
            if result.status == "failed" and not continue_on_error:
                for ref in pending:
                    ray.cancel(ref, force=True)
                return finish()
        drain()
        return finish()


def run_cellvit_isolated(
    wsi_path: str,
    *,
    device: str = "cuda",
    image_mpp: float | None = None,
    image_magnification: int | None = None,
    available_cpus: int | None = None,
    available_memory_mb: int | None = None,
    worker_log_path: str | Path | None = None,
) -> Any:
    """Run ``segment_cells`` in-process, or in an isolated child process when inside a Ray worker.

    CellViT starts its own internal Ray runtime; inside a Ray worker that would
    collide with the outer cluster, so the segmentation runs in a subprocess
    with the outer ``RAY_*`` environment stripped and its output appended to
    ``worker_log_path``.
    """
    import ray

    from helios.components.cells import segment_cells

    kwargs: dict[str, Any] = {
        "device": device,
        "image_mpp": image_mpp,
        "image_magnification": image_magnification,
        "available_cpus": available_cpus,
        "available_memory_mb": available_memory_mb,
    }
    if not ray.is_initialized():
        return segment_cells(wsi_path, **kwargs)

    import pickle
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        out_pkl = Path(tmp.name)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAY_")}
    code = (
        "import pickle; from helios.components.cells import segment_cells; "
        f"df = segment_cells({wsi_path!r}, **{kwargs!r}); "
        f"pickle.dump(df, open({str(out_pkl)!r}, 'wb'))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
        if worker_log_path is not None:
            with Path(worker_log_path).open("a") as logf:
                logf.write(proc.stdout)
                logf.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Isolated CellViT process failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
        with out_pkl.open("rb") as f:
            return pickle.load(f)
    finally:
        out_pkl.unlink(missing_ok=True)
