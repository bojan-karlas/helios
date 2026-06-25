"""Component execution context.

Every component stub receives a :class:`ComponentContext` plus a ``keys`` dict
of its partition coordinates for one invocation (e.g. ``{"fold": 0,
"size_mm": 0.25, "model": "virchow2"}``). The context bundles the run config and
the runtime seams a component needs: the artifact store (id+key IO), the path
resolver, the progress reporter, the logger, the resolved column roles, and the
model-bundle root for persisting/loading fitted models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helios.config.models import RunConfig
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver
from helios.progress.reporter import ProgressReporter
from helios.registry.bundle import Bundle


@dataclass
class ComponentContext:
    config: RunConfig
    store: ArtifactStore
    resolver: Resolver
    reporter: ProgressReporter
    logger: logging.Logger
    roles: dict[str, Any]

    def bundle(self, component: str) -> Bundle:
        """Model bundle for a component under the configured models root."""
        models_root = Path(self.resolver.roots._role("models"))
        return Bundle(root=models_root, version=self.config.version, component=component)
