"""Config generation + safety tests.

Guards two invariants:
* ``configs/default.yaml`` is in sync with the stage signatures (run
  ``make configs`` if this fails).
* Committed configs never reference a real dataset (no ``dataset:`` block).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from helios.config.introspect import build_default_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
DEFAULT_YAML = CONFIGS_DIR / "default.yaml"


def test_default_config_is_in_sync() -> None:
    committed = yaml.safe_load(DEFAULT_YAML.read_text())
    assert committed == build_default_config(), (
        "configs/default.yaml is stale; run `make configs` to regenerate it."
    )


def test_committed_configs_have_no_dataset_block() -> None:
    for path in CONFIGS_DIR.rglob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            assert "dataset" not in data, (
                f"{path.relative_to(ROOT)} must not reference a dataset; "
                "dataset blocks belong only in gitignored user --config files."
            )
