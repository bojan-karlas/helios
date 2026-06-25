"""Run progress reporting: status.json + events.jsonl.

Matches the Docker mount contract (a writable status/log location). The reporter
emits a single rolling ``status.json`` (latest state) and an append-only
``events.jsonl`` (one JSON object per line) so a host/web-app can watch a run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProgressReporter:
    out_dir: Path
    status_name: str = "status.json"
    events_name: str = "events.jsonl"
    _state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def status_path(self) -> Path:
        return self.out_dir / self.status_name

    @property
    def events_path(self) -> Path:
        return self.out_dir / self.events_name

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self.events_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def update(self, **fields: Any) -> None:
        self._state.update(fields)
        self._state["ts"] = time.time()
        self.status_path.write_text(json.dumps(self._state, indent=2, default=str))

    def stage(self, component: str, status: str, **fields: Any) -> None:
        """Convenience: record a component's status as both an event and state."""
        self.event("stage", component=component, status=status, **fields)
        self.update(current_component=component, current_status=status)
