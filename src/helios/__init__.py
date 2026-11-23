"""HELIOS -- computational-pathology pipeline for melanoma recurrence-risk prediction."""

import os

__version__ = "0.1.0"

# Under ``uv run`` Ray would otherwise zip and ship the whole working directory
# (checkpoints included) to every worker; helios is installed in the workers'
# environment, so opt out before Ray is imported anywhere. Set explicitly to override.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
# numpy's transparent huge pages stall large allocations in kernel compaction when memory is fragmented.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
