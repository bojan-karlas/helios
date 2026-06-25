"""Column roles + data-dictionary binding.

Roles map abstract pipeline inputs (targets, concepts, split key, tiling fields)
to concrete ``image_metadata`` column names via a roles YAML, so the same code
runs on cohorts with different column names. The data dictionary
(``image_metadata.datadict.yaml``) is the canonical, versioned column contract
used for validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from helios.data.catalog import _configs_dir


def load_roles(roles_path: str | Path | None = None) -> dict[str, Any]:
    if roles_path is None:
        roles_path = _configs_dir() / "roles" / "default.yaml"
    data = yaml.safe_load(Path(roles_path).read_text())
    return data or {}


def load_datadict(datadict_path: str | Path | None = None) -> dict[str, Any]:
    if datadict_path is None:
        datadict_path = _configs_dir() / "schema" / "image_metadata.datadict.yaml"
    data = yaml.safe_load(Path(datadict_path).read_text())
    return data or {}


def resolve_role(roles: dict[str, Any], role: str) -> Any:
    """Resolve a role name to its bound column(s); raises if unbound."""
    if role not in roles:
        raise KeyError(f"role {role!r} is not bound in roles config")
    return roles[role]
