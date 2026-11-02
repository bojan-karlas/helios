"""Epoch and phase feedback for slide-level training."""
from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from time import monotonic
from typing import TypeVar

from tqdm import tqdm

from helios.progress.progress import Progress

T = TypeVar("T")


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    def __init__(self, progress: Progress, epochs: int, prefix: str) -> None:
        self.progress = progress
        self.epochs = epochs
        self.prefix = prefix
        self.completed = 0
        self.elapsed = 0.0
        self.epoch_started = monotonic()
        self.loss_sum = 0.0
        self.steps = 0
        self.skipped = 0
        self.validation_skipped = 0

    def budget_eta(self) -> str:
        if not self.completed:
            return "estimating"
        remaining = self.elapsed / self.completed * (self.epochs - self.completed)
        remaining -= monotonic() - self.epoch_started
        return duration(remaining)

    @contextmanager
    def display(self) -> Iterator[TrainingProgress]:
        enabled = self.progress.enabled and self.progress.use_tqdm and sys.stderr.isatty()
        with tqdm(
            total=self.epochs, desc=f"{self.prefix}Epochs", disable=not enabled, position=1, dynamic_ncols=True,
        ) as bar:
            self.bar = bar
            yield self

    def start_epoch(self) -> None:
        self.epoch_started = monotonic()
        self.loss_sum = 0.0
        self.steps = 0
        self.skipped = 0
        self.validation_skipped = 0
        self.progress.log(
            f"{self.prefix}epoch={self.completed + 1}/{self.epochs} starting "
            f"ETA to {self.epochs} epochs={self.budget_eta()}"
        )

    def record_loss(self, loss: float) -> None:
        self.loss_sum += loss
        self.steps += 1

    @property
    def mean_loss(self) -> str:
        return f"{self.loss_sum / self.steps:.4f}" if self.steps else "n/a"

    def phase(self, items: Sequence[T], name: str) -> Iterator[T]:
        desc = f"{self.prefix}Epoch {self.completed + 1}/{self.epochs} {name}"
        started = last_log = monotonic()
        enabled = self.progress.enabled and self.progress.use_tqdm and sys.stderr.isatty()
        if not enabled:
            self.progress.log(f"{desc} 0/{len(items)} slides")
        with tqdm(
            total=len(items), desc=desc, unit="slide", disable=not enabled,
            position=2, leave=False, dynamic_ncols=True,
        ) as bar:
            for count, item in enumerate(items, 1):
                yield item
                bar.update()
                now = monotonic()
                if now - last_log >= (1 if enabled else 30) or count == len(items):
                    elapsed = now - started
                    eta = elapsed / count * (len(items) - count)
                    details = (
                        f"loss={self.mean_loss} "
                        f"skipped={self.skipped if name == 'train' else self.validation_skipped}"
                    )
                    bar.set_postfix_str(details)
                    self.bar.set_postfix_str(f"ETA to {self.epochs} epochs={self.budget_eta()}")
                    if not enabled:
                        self.progress.log(
                            f"{desc} {count}/{len(items)} slides "
                            f"rate={count / max(elapsed, 1e-9):.2f}/s "
                            f"elapsed={duration(elapsed)} phase_ETA={duration(eta)} {details} "
                            f"ETA to {self.epochs} epochs={self.budget_eta()}"
                        )
                    last_log = now

    def finish_epoch(self) -> float:
        elapsed = monotonic() - self.epoch_started
        self.elapsed += elapsed
        self.completed += 1
        self.epoch_started = monotonic()
        self.bar.update()
        self.bar.set_postfix_str(f"ETA to {self.epochs} epochs={self.budget_eta()}")
        return elapsed
