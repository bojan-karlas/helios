"""Workaround for a torch/faiss OpenMP conflict on macOS: faiss-cpu and torch
each bundle their own OpenMP runtime, and having both loaded in one process
segfaults inside whichever library runs its threaded ops second (faiss.Kmeans
or torch's LayerNorm) regardless of import order. KMP_DUPLICATE_LIB_OK lets
both runtimes coexist; OMP_NUM_THREADS=1 avoids the thread-pool clash that
still segfaults with duplicate-lib-ok alone. Must be set before either library
is imported anywhere in the test session, hence here in conftest.py.
"""
from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: E402, F401


@pytest.fixture(autouse=True)
def restore_helios_logging():
    logger = logging.getLogger("helios")
    handlers, propagate, level = logger.handlers[:], logger.propagate, logger.level
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.propagate = propagate
    logger.setLevel(level)
