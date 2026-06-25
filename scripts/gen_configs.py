#!/usr/bin/env python3
"""Generate ``configs/default.yaml`` from stage signatures.

Code is the source of truth: each stage declares its tunable parameters as typed
keyword defaults. This script introspects them (see
``helios.config.introspect``) and writes the committed default config, keyed by
stage. A test (``tests/test_configs.py``) asserts the committed file matches what
this script would regenerate, so the two can never drift.

Run after changing any stage default:  python scripts/gen_configs.py
(or ``make configs``).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from helios.config.introspect import build_default_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "default.yaml"

HEADER = (
    "# GENERATED from stage signatures — do not hand-edit.\n"
    "# Keyed by verb then noun; entries are that stage's tunable parameters\n"
    "# (1:1 with `helios <verb> <noun> --help`). Change defaults in the stage\n"
    "# functions under src/helios/stages/, then run `make configs`.\n"
)


def main() -> None:
    config = build_default_config()
    dumper = yaml.SafeDumper
    dumper.ignore_aliases = lambda self, data: True  # no &anchors for shared list defaults
    body = yaml.dump(config, Dumper=dumper, sort_keys=True, default_flow_style=False)
    OUT.write_text(HEADER + body)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
