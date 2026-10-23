"""Logging setup (run.log + per-worker logs + run-id generation)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def normalize_run_tag(tag: str | None) -> str | None:
    """Normalize an optional user tag to ASCII kebab-case."""
    if tag is None:
        return None
    ascii_text = tag.encode("ascii", errors="ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if not slug:
        raise ValueError(f"Run tag {tag!r} contains no valid ASCII alphanumeric characters.")
    return slug


def generate_run_id(tag: str | None = None, *, now: datetime | None = None) -> str:
    """Format a UTC timestamp run ID: ``YYYY-MM-DD-HH-MM-SS.ffffff-utc[-<tag>]``."""
    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = ts.strftime("%Y-%m-%d-%H-%M-%S.%f-utc")
    slug = normalize_run_tag(tag)
    return f"{stamp}-{slug}" if slug else stamp


def create_run_log_dir(base_dir: str | Path, tag: str | None = None) -> tuple[str, Path]:
    """Create a unique, absolute ``<base_dir>/<run_id>`` directory (and ``workers/`` child).

    The path is made absolute so remote workers with a different cwd resolve it.
    """
    base = Path(base_dir).absolute()
    base.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = generate_run_id(tag)
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            (run_dir / "workers").mkdir(parents=True, exist_ok=True)
            return run_id, run_dir
        except FileExistsError:
            continue


def get_logger(name: str = "helios") -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("helios")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "workers").mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


@contextmanager
def worker_log(
    run_log_dir: str | Path | None,
    worker_id: str,
    *,
    level: int = logging.INFO,
) -> Iterator[tuple[logging.Logger, Path | None]]:
    """A dedicated file logger for one worker under ``<run_log_dir>/workers/``.

    Yields ``(logger, log_path)``; the file handler is closed on exit so repeated
    use (one unit after another in the same process) never leaks descriptors.
    With ``run_log_dir=None`` the logger has no handlers and ``log_path`` is ``None``.
    """
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", worker_id).strip("_") or "worker"
    logger = logging.getLogger(f"helios.worker.{safe_id}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    if run_log_dir is None:
        logger.addHandler(logging.NullHandler())
        yield logger, None
        return

    workers_dir = Path(run_log_dir) / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    log_path = workers_dir / f"{safe_id}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    logger.addHandler(handler)
    try:
        yield logger, log_path
    finally:
        logger.removeHandler(handler)
        handler.close()
