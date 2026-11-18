"""Code version of the running HELIOS process, captured once when this module is imported.

Capturing at import matches the code the process actually runs: editing or committing
files later does not change modules that are already loaded.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args], capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _capture() -> dict[str, object] | None:
    """``{"commit", "dirty"}`` for a git checkout that tracks this package, else ``None``."""
    if _git("ls-files", "--error-unmatch", "__init__.py") is None:
        return None
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain")
    return {"commit": commit.strip(), "dirty": None if status is None else bool(status.strip())}


#: ``{"commit": <sha>, "dirty": <non-ignored files differ from the commit>}``, or ``None`` outside a git checkout.
CODE_VERSION = _capture()
