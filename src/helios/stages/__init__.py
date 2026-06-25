"""Stages — CLI-callable orchestrators that wire components to the store.

A *stage* is one ``verb noun`` cell (e.g. ``prep tile-features``, ``fit mil``):
it groups one or more components into a unit of work you can run on its own
(``helios prep tile-features``) or import and call
(``from helios.stages.tile_features import prep_tile_features``).

Unlike components, stages are NOT pure: they own all artifact-store I/O, the
dataset / partition / fold loops, per-unit skip logic (``--force``), selector
validation, and Progress wiring.

Modules are organised by *noun* (domain area); functions are named ``verb_noun``
so they read unambiguously and grep cleanly. The :data:`STAGES` registry maps
``verb -> noun -> function`` (nouns use the hyphenated CLI spelling); the CLI and
the config introspector both iterate it, so a stage is reachable and
configurable the moment it is registered.

Each stage's keyword parameters with defaults are the single source of truth for
its configuration: ``scripts/gen_configs.py`` introspects them to generate
``configs/default.yaml`` (run ``make configs`` after changing a default).
Framework seams (``datasets``, ``image_ids``, ``force``, ``progress``) are
runtime arguments, not config keys.
"""
from __future__ import annotations

from collections.abc import Callable

from helios.stages import (
    augment,
    cell_features,
    cells,
    concept,
    mil,
    morphology,
    report,
    risk,
    splits,
    thumbnails,
    tile_features,
    tiles,
)

#: Registry of runnable stages: ``verb -> noun -> stage function``.
STAGES: dict[str, dict[str, Callable[..., None]]] = {
    "prep": {
        "splits": splits.prep_splits,
        "thumbnails": thumbnails.prep_thumbnails,
        "tiles": tiles.prep_tiles,
        "augment": augment.prep_augment,
        "tile-features": tile_features.prep_tile_features,
        "cells": cells.prep_cells,
        "cell-features": cell_features.prep_cell_features,
    },
    "fit": {
        "mil": mil.fit_mil,
        "concept": concept.fit_concept,
        "morphology": morphology.fit_morphology,
        "risk": risk.fit_risk,
    },
    "predict": {
        "mil": mil.predict_mil,
        "concept": concept.predict_concept,
        "morphology": morphology.predict_morphology,
        "risk": risk.predict_risk,
    },
    "report": {
        "build": report.build_report,
    },
}

__all__ = [
    "STAGES",
    "augment",
    "cell_features",
    "cells",
    "concept",
    "mil",
    "morphology",
    "report",
    "risk",
    "splits",
    "thumbnails",
    "tile_features",
    "tiles",
]
