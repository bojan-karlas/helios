"""Unified progress + logging for components and stages.

``Progress`` is the single object threaded through the pipeline for user-facing
feedback. It wraps a tqdm progress bar and message logging, and can optionally
forward events to a :class:`ProgressReporter` (status.json + events.jsonl) so a
host/web-app sees the same activity.

Design (see CONTRIBUTING): components take ``progress: Progress = DEFAULT_PROGRESS``
as their *only* framework-ish argument. ``DEFAULT_PROGRESS`` is a shared no-op,
so a component called standalone (a test, a notebook) is silent by default. A
**stage** constructs a real, reporter-backed ``Progress`` and passes it down, so
the decision to show bars / write status files lives in the orchestration layer,
never hidden inside the pure component.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from helios.progress.reporter import ProgressReporter

T = TypeVar("T")

_LOG = logging.getLogger("helios")


class Progress:
    """A progress bar + message sink, optionally backed by a reporter.

    Parameters
    ----------
    reporter:
        Optional :class:`ProgressReporter`; when set, ``log`` and task lifecycle
        events are also appended to ``events.jsonl`` / rolled into ``status.json``.
    use_tqdm:
        Render a tqdm bar for :meth:`task`. Disable for non-interactive runs.
    logger:
        Logger for :meth:`log`. Defaults to the ``helios`` logger.
    enabled:
        When ``False`` the instance is a no-op (this is ``DEFAULT_PROGRESS``).
    """

    def __init__(
        self,
        *,
        reporter: ProgressReporter | None = None,
        use_tqdm: bool = True,
        logger: logging.Logger | None = None,
        enabled: bool = True,
    ) -> None:
        self.reporter = reporter
        self.use_tqdm = use_tqdm
        self.logger = logger or _LOG
        self.enabled = enabled

    # -- constructors ---------------------------------------------------------

    @classmethod
    def console(cls, *, reporter: ProgressReporter | None = None) -> Progress:
        """A visible, console-backed progress (tqdm bar + logging)."""
        return cls(reporter=reporter, use_tqdm=True, enabled=True)

    @classmethod
    def null(cls) -> Progress:
        """A no-op progress: no bar, no logging, no reporter."""
        return cls(use_tqdm=False, enabled=False)

    # -- messages -------------------------------------------------------------

    def log(self, message: str, *, level: int = logging.INFO) -> None:
        if not self.enabled:
            return
        self.logger.log(level, message)
        if self.reporter is not None:
            self.reporter.event("log", level=logging.getLevelName(level), message=message)

    # -- iteration / bars -----------------------------------------------------

    def task(self, iterable: Iterable[T], *, total: int | None = None, desc: str = "") -> Iterator[T]:
        """Iterate ``iterable`` while showing a bar and emitting start/done events."""
        if not self.enabled:
            yield from iterable
            return

        if total is None:
            total = _safe_len(iterable)
        if self.reporter is not None:
            self.reporter.event("task_start", desc=desc, total=total)

        n = 0
        for item in self._wrap(iterable, total=total, desc=desc):
            yield item
            n += 1

        if self.reporter is not None:
            self.reporter.event("task_done", desc=desc, count=n)

    def subtask(self, desc: str) -> Progress:
        """A child progress sharing the same reporter (for nested work)."""
        if not self.enabled:
            return self
        child = Progress(reporter=self.reporter, use_tqdm=self.use_tqdm, logger=self.logger)
        child.log(desc)
        return child

    # -- internals ------------------------------------------------------------

    def _wrap(self, iterable: Iterable[T], *, total: int | None, desc: str) -> Iterable[T]:
        if not self.use_tqdm:
            return iterable
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return iterable
        return tqdm(iterable, total=total, desc=desc)


def _safe_len(iterable: Iterable[object]) -> int | None:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


#: Shared no-op progress used as the default for pure components.
DEFAULT_PROGRESS = Progress.null()
